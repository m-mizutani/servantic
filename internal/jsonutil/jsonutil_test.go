package jsonutil_test

import (
	"encoding/json"
	"testing"

	"github.com/gollem-dev/gollem/internal/jsonutil"
	"github.com/m-mizutani/gt"
)

func TestDecodeObjectPreservesWideIntegers(t *testing.T) {
	type testCase struct {
		input string
		key   string
		// want is the exact JSON literal the value must produce when re-encoded.
		want string
	}

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			decoded, err := jsonutil.DecodeObject([]byte(tc.input))
			gt.NoError(t, err)

			encoded, err := json.Marshal(decoded[tc.key])
			gt.NoError(t, err)
			gt.Equal(t, tc.want, string(encoded))
		}
	}

	t.Run("integer beyond float64 precision", runTest(testCase{
		input: `{"id":9007199254740993}`,
		key:   "id",
		want:  "9007199254740993",
	}))

	t.Run("large signed integer", runTest(testCase{
		input: `{"id":1234567890123456789}`,
		key:   "id",
		want:  "1234567890123456789",
	}))

	t.Run("integer wider than int64", runTest(testCase{
		input: `{"id":123456789012345678901234567890}`,
		key:   "id",
		want:  "123456789012345678901234567890",
	}))

	t.Run("small integer stays float64", runTest(testCase{
		input: `{"n":42}`,
		key:   "n",
		want:  "42",
	}))

	t.Run("fractional number", runTest(testCase{
		input: `{"n":1.5}`,
		key:   "n",
		want:  "1.5",
	}))
}

// A number that a float64 represents exactly keeps the float64 type, so tools that assert
// args["n"].(float64) keep working for every value they work for today.
func TestDecodeObjectKeepsFloat64ForExactValues(t *testing.T) {
	decoded, err := jsonutil.DecodeObject([]byte(`{"a":42,"b":3.0,"c":-7,"d":1e21}`))
	gt.NoError(t, err)

	gt.Equal(t, float64(42), gt.Cast[float64](t, decoded["a"]))
	gt.Equal(t, float64(3), gt.Cast[float64](t, decoded["b"]))
	gt.Equal(t, float64(-7), gt.Cast[float64](t, decoded["c"]))
	gt.Equal(t, float64(1e21), gt.Cast[float64](t, decoded["d"]))
}

func TestDecodeObjectPreservesNestedNumbers(t *testing.T) {
	decoded, err := jsonutil.DecodeObject([]byte(`{"outer":{"ids":[9007199254740993,1]}}`))
	gt.NoError(t, err)

	encoded, err := json.Marshal(decoded)
	gt.NoError(t, err)
	gt.Equal(t, `{"outer":{"ids":[9007199254740993,1]}}`, string(encoded))
}

// Decode targets a concrete struct whose map field holds the interface values that
// UseNumber turns into json.Number, so the walk has to descend through the struct.
func TestDecodeIntoStructWithMapField(t *testing.T) {
	type payload struct {
		Response map[string]any `json:"response"`
	}

	var p payload
	gt.NoError(t, jsonutil.Decode([]byte(`{"response":{"id":9007199254740993,"n":2}}`), &p))

	encoded, err := json.Marshal(p.Response)
	gt.NoError(t, err)
	gt.Equal(t, `{"id":9007199254740993,"n":2}`, string(encoded))
	gt.Equal(t, float64(2), gt.Cast[float64](t, p.Response["n"]))
}

func TestDecodeObjectRejectsNonObject(t *testing.T) {
	_, err := jsonutil.DecodeObject([]byte(`[1,2,3]`))
	gt.Error(t, err)
}

// json.Decoder reads one value and ignores the rest, where json.Unmarshal reports it.
// Callers depend on the rejection: a tool result that is an object followed by prose has
// to fall back to being carried as raw text instead of being truncated to the object.
func TestDecodeObjectRejectsTrailingData(t *testing.T) {
	_, err := jsonutil.DecodeObject([]byte(`{"a":1} and then some prose`))
	gt.Error(t, err)
}

// A []byte field is a slice of uint8 and can never hold a json.Number, so the walk must
// not descend into it. This pins the behaviour; the cost shows up on image and PDF
// content, whose Data field is megabytes of []byte.
func TestDecodeSkipsByteSlices(t *testing.T) {
	type payload struct {
		Data []byte         `json:"data"`
		Meta map[string]any `json:"meta"`
	}

	var p payload
	gt.NoError(t, jsonutil.Decode([]byte(`{"data":"aGVsbG8=","meta":{"id":9007199254740993}}`), &p))

	gt.Equal(t, "hello", string(p.Data))
	encoded, err := json.Marshal(p.Meta)
	gt.NoError(t, err)
	gt.Equal(t, `{"id":9007199254740993}`, string(encoded))
}

func BenchmarkDecodeLargeByteSlice(b *testing.B) {
	type payload struct {
		Data []byte `json:"data"`
	}

	blob := make([]byte, 1<<20)
	raw, err := json.Marshal(payload{Data: blob})
	if err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for b.Loop() {
		var p payload
		if err := jsonutil.Decode(raw, &p); err != nil {
			b.Fatal(err)
		}
	}
}

func TestMarshalIndentNoEscape(t *testing.T) {
	out, err := jsonutil.MarshalIndentNoEscape(map[string]any{"description": "a<b && c>d"}, "", "  ")
	gt.NoError(t, err)

	gt.Equal(t, "{\n  \"description\": \"a<b && c>d\"\n}", string(out))
}
