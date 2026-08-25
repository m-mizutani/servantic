// Package jsonutil provides JSON decoding helpers that preserve the exact value of
// every number in the input.
//
// encoding/json decodes every JSON number into a float64 when the destination is an
// interface value. Tool arguments, tool results and history contents all travel through
// map[string]any on their way to and from the provider, so an integer wider than 53 bits
// (an account ID, a nanosecond timestamp) is silently rounded before the model ever sees
// it. The helpers here decode with json.Decoder.UseNumber and then hand back a float64
// for every number a float64 can hold exactly, keeping json.Number only for the values
// that would otherwise be corrupted. Code that type-asserts float64 therefore keeps
// working for every value it works for today.
//
// This package must not import the root gollem package: internal/convert already depends
// on gollem, and the root package needs these helpers too.
package jsonutil

import (
	"bytes"
	"encoding/json"
	"reflect"
	"strconv"
	"strings"
	"sync"

	"github.com/m-mizutani/goerr/v2"
)

// DecodeObject decodes a JSON object into a map, preserving numbers that a float64
// cannot represent exactly. It returns an error when data is not a JSON object.
func DecodeObject(data []byte) (map[string]any, error) {
	var m map[string]any
	if err := Decode(data, &m); err != nil {
		return nil, err
	}
	return m, nil
}

// Decode unmarshals data into v, preserving numbers that a float64 cannot represent
// exactly. Numbers decoded into an interface value become float64 as usual unless the
// conversion would change the value, in which case they stay json.Number.
func Decode(data []byte, v any) error {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	if err := dec.Decode(v); err != nil {
		return err
	}
	// Decoder.Decode reads one value and ignores whatever follows, where json.Unmarshal
	// reports it. Callers rely on that rejection: a tool result that is a JSON object
	// followed by prose must fall back to being carried as raw text, not be silently
	// truncated to its first object.
	if dec.More() {
		return goerr.New("unexpected data after top-level JSON value")
	}
	normalizeInPlace(v)
	return nil
}

var numberType = reflect.TypeOf(json.Number(""))

var (
	canHoldNumberCacheMu sync.RWMutex
	canHoldNumberCache   = map[reflect.Type]bool{}
)

// canHoldNumber reports whether a value of type t could contain a json.Number anywhere
// inside it. Only an interface-typed field can hold one, so a type with no interface in
// its shape needs no walk at all. Without this check the walk descends into every element
// of a []byte, which makes decoding an image or a PDF attachment cost one reflect call per
// byte.
func canHoldNumber(t reflect.Type) bool {
	canHoldNumberCacheMu.RLock()
	cached, ok := canHoldNumberCache[t]
	canHoldNumberCacheMu.RUnlock()
	if ok {
		return cached
	}

	result := computeCanHoldNumber(t, map[reflect.Type]bool{})

	canHoldNumberCacheMu.Lock()
	canHoldNumberCache[t] = result
	canHoldNumberCacheMu.Unlock()
	return result
}

// computeCanHoldNumber walks the type graph. visiting guards against a recursive type,
// whose cycle cannot introduce an interface that the rest of the walk does not already see.
func computeCanHoldNumber(t reflect.Type, visiting map[reflect.Type]bool) bool {
	if t == numberType || t.Kind() == reflect.Interface {
		return true
	}
	if visiting[t] {
		return false
	}
	visiting[t] = true
	defer delete(visiting, t)

	switch t.Kind() {
	case reflect.Pointer, reflect.Slice, reflect.Array:
		return computeCanHoldNumber(t.Elem(), visiting)
	case reflect.Map:
		return computeCanHoldNumber(t.Elem(), visiting)
	case reflect.Struct:
		for i := 0; i < t.NumField(); i++ {
			if !t.Field(i).IsExported() {
				continue
			}
			if computeCanHoldNumber(t.Field(i).Type, visiting) {
				return true
			}
		}
	}
	return false
}

// normalizeInPlace rewrites every json.Number reachable from v that a float64 represents
// exactly. UseNumber only produces json.Number where the destination is an interface
// value, so the walk looks for interface-typed values wherever they can occur: at the top
// level, inside maps and slices, and in the exported fields of a decoded struct such as
// ToolResponseContent, whose Response is a map[string]any.
func normalizeInPlace(v any) {
	normalizeReflect(reflect.ValueOf(v))
}

func normalizeReflect(v reflect.Value) {
	if !v.IsValid() || !canHoldNumber(v.Type()) {
		return
	}

	switch v.Kind() {
	case reflect.Pointer:
		if !v.IsNil() {
			normalizeReflect(v.Elem())
		}

	case reflect.Interface:
		if v.IsNil() {
			return
		}
		inner := v.Elem()
		if inner.Type() == numberType && v.CanSet() {
			v.Set(reflect.ValueOf(normalizeNumber(inner.Interface().(json.Number))))
			return
		}
		normalizeReflect(inner)

	case reflect.Map:
		for _, key := range v.MapKeys() {
			elem := v.MapIndex(key)
			// A map element is not addressable, so a json.Number has to be written back
			// through SetMapIndex. Maps and slices nested inside it share their backing
			// storage with the copy, so those are normalized in place.
			if elem.Kind() == reflect.Interface && !elem.IsNil() {
				inner := elem.Elem()
				if inner.Type() == numberType {
					v.SetMapIndex(key, reflect.ValueOf(normalizeNumber(inner.Interface().(json.Number))))
					continue
				}
				normalizeReflect(inner)
				continue
			}
			normalizeReflect(elem)
		}

	case reflect.Slice, reflect.Array:
		for i := 0; i < v.Len(); i++ {
			normalizeReflect(v.Index(i))
		}

	case reflect.Struct:
		t := v.Type()
		for i := 0; i < v.NumField(); i++ {
			if !t.Field(i).IsExported() {
				continue
			}
			normalizeReflect(v.Field(i))
		}
	}
}

// normalizeNumber returns the float64 form of n when that conversion is lossless, and n
// itself otherwise. A non-integer literal is always converted: float64 is the widest type
// encoding/json would have produced for it anyway, so keeping json.Number there would
// change the type of values that are handled correctly today without fixing anything.
func normalizeNumber(n json.Number) any {
	s := n.String()

	if strings.ContainsAny(s, ".eE") {
		f, err := n.Float64()
		if err != nil {
			return n
		}
		return f
	}

	i, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		// Outside int64: a float64 cannot hold it exactly either.
		return n
	}
	if int64(float64(i)) != i {
		return n
	}
	return float64(i)
}

// MarshalIndentNoEscape marshals v as indented JSON without escaping the three
// HTML-significant characters "<", ">" and "&". encoding/json replaces them with
// their six-character unicode escapes by default, which is invisible to a JSON
// parser but not to a reader: this output is embedded verbatim into a system
// prompt, where the model reads the escape sequences instead of the characters
// the schema author wrote.
func MarshalIndentNoEscape(v any, prefix, indent string) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	enc.SetIndent(prefix, indent)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	// Encoder.Encode appends a newline that MarshalIndent does not produce.
	return bytes.TrimRight(buf.Bytes(), "\n"), nil
}
