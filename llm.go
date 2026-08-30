package gollem

import (
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"log/slog"

	"github.com/m-mizutani/goerr/v2"
)

// LLMClient is a client for each LLM service.
type LLMClient interface {
	NewSession(ctx context.Context, options ...SessionOption) (Session, error)
	GenerateEmbedding(ctx context.Context, dimension int, input []string) ([][]float64, error)
}

// ModelNamer is implemented by an LLMClient that can report the model it
// generates through. The value is the model name the client was configured
// with, not the model id an API response may report.
//
// It is optional: an LLMClient that does not implement it keeps working, and a
// caller obtains the name with a type assertion.
//
//	if n, ok := client.(gollem.ModelNamer); ok {
//		model = n.Model()
//	}
type ModelNamer interface {
	Model() string
}

type FunctionCall struct {
	ID        string
	Name      string
	Arguments map[string]any
}

// Response is a general response type for each gollem.
type Response struct {
	Texts         []string
	Thoughts      []string
	FunctionCalls []*FunctionCall
	InputToken    int
	OutputToken   int

	// CacheCreationInputToken is the number of input tokens written to the
	// prompt cache on this call. Only Claude reports writes; 0 elsewhere.
	CacheCreationInputToken int
	// CacheReadInputToken is the number of input tokens served from the prompt
	// cache (cache hits) on this call. Reported by all providers when caching
	// occurs. InputToken always counts total input (including these).
	CacheReadInputToken int

	// Error is an error that occurred during the generation for streaming response.
	Error error
}

func (r *Response) HasData() bool {
	return len(r.Texts) > 0 || len(r.Thoughts) > 0 || len(r.FunctionCalls) > 0 || r.Error != nil
}

type Input interface {
	isInput() restrictedValue
	LogValue() slog.Value
	String() string
}

type restrictedValue struct{}

// Text is a text input as prompt.
// Usage:
// input := gollem.Text("Hello, world!")
type Text string

func (t Text) isInput() restrictedValue {
	return restrictedValue{}
}

func (t Text) LogValue() slog.Value {
	return slog.StringValue(string(t))
}

func (t Text) String() string {
	return string(t)
}

// FunctionResponse is a function response.
// Usage:
//
//	input := gollem.FunctionResponse{
//		Name:      "function_name",
//		Arguments: map[string]any{"key": "value"},
//	}
type FunctionResponse struct {
	ID    string
	Name  string
	Data  map[string]any
	Error error
}

func (f FunctionResponse) isInput() restrictedValue {
	return restrictedValue{}
}

// String returns a string representation of the FunctionResponse
func (f FunctionResponse) String() string {
	if f.Error != nil {
		return f.Name + " (error: " + f.Error.Error() + ")"
	}
	return f.Name + " (success)"
}

// LogValue returns a slog.Value for the FunctionResponse
func (f FunctionResponse) LogValue() slog.Value {
	attrs := []slog.Attr{
		slog.String("id", f.ID),
		slog.String("name", f.Name),
	}

	if f.Data != nil {
		attrs = append(attrs, slog.Any("data", f.Data))
	}

	if f.Error != nil {
		attrs = append(attrs, slog.String("error", f.Error.Error()))
	}

	return slog.GroupValue(attrs...)
}

// ImageMimeType represents supported MIME types for images
type ImageMimeType string

const (
	ImageMimeTypeJPEG ImageMimeType = "image/jpeg"
	ImageMimeTypePNG  ImageMimeType = "image/png"
	ImageMimeTypeGIF  ImageMimeType = "image/gif" // Claude, OpenAI only
	ImageMimeTypeWebP ImageMimeType = "image/webp"
	// Gemini-specific formats
	ImageMimeTypeHEIC ImageMimeType = "image/heic" // Gemini only
	ImageMimeTypeHEIF ImageMimeType = "image/heif" // Gemini only
)

// Image represents an image input for LLM
type Image struct {
	data     []byte
	mimeType ImageMimeType
}

// isInput implements Input interface
func (i Image) isInput() restrictedValue {
	return restrictedValue{}
}

func (i Image) LogValue() slog.Value {
	return slog.StringValue(i.String())
}

func (i Image) String() string {
	return fmt.Sprintf("image (%d bytes, %s)", len(i.data), i.mimeType)
}

// Data returns the image data as bytes
func (i Image) Data() []byte {
	return i.data
}

// MimeType returns the MIME type of the image
func (i Image) MimeType() string {
	return string(i.mimeType)
}

// Base64 returns the base64 encoded string of the image data
func (i Image) Base64() string {
	return base64.StdEncoding.EncodeToString(i.data)
}

// imageMimeEntry maps a MIME type to a byte-pattern matcher used for detection.
type imageMimeEntry struct {
	mimeType ImageMimeType
	match    func(data []byte) bool
}

// imageMimeTable is the single source of truth for supported image MIME types.
// Both detectImageMimeType and IsValidImageMimeType derive their behavior from it.
var imageMimeTable = []imageMimeEntry{
	{
		ImageMimeTypeJPEG,
		func(d []byte) bool { return d[0] == 0xFF && d[1] == 0xD8 && d[2] == 0xFF },
	},
	{
		ImageMimeTypePNG,
		func(d []byte) bool {
			return bytes.Equal(d[:8], []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A})
		},
	},
	{
		// GIF87a and GIF89a share the same MIME type.
		ImageMimeTypeGIF,
		func(d []byte) bool {
			return bytes.Equal(d[:6], []byte("GIF87a")) || bytes.Equal(d[:6], []byte("GIF89a"))
		},
	},
	{
		// WebP: "RIFF" at offset 0, "WEBP" at offset 8.
		ImageMimeTypeWebP,
		func(d []byte) bool {
			return bytes.Equal(d[:4], []byte("RIFF")) && len(d) >= 12 && bytes.Equal(d[8:12], []byte("WEBP"))
		},
	},
	{
		// HEIC brands: heic, heix, heim, heis.
		ImageMimeTypeHEIC,
		func(d []byte) bool {
			if !bytes.Equal(d[4:8], []byte("ftyp")) {
				return false
			}
			brand := string(d[8:12])
			return brand == "heic" || brand == "heix" || brand == "heim" || brand == "heis"
		},
	},
	{
		// HEIF brands: heif, mif1.
		ImageMimeTypeHEIF,
		func(d []byte) bool {
			if !bytes.Equal(d[4:8], []byte("ftyp")) {
				return false
			}
			brand := string(d[8:12])
			return brand == "heif" || brand == "mif1"
		},
	},
}

// detectImageMimeType detects MIME type from image data
func detectImageMimeType(data []byte) (ImageMimeType, error) {
	if len(data) < 12 {
		return "", goerr.New("data too short to detect format")
	}
	for _, e := range imageMimeTable {
		if e.match(data) {
			return e.mimeType, nil
		}
	}
	return "", goerr.New("unsupported image format")
}

// IsValidImageMimeType checks if the MIME type is supported
func IsValidImageMimeType(mimeType ImageMimeType) bool {
	for _, e := range imageMimeTable {
		if e.mimeType == mimeType {
			return true
		}
	}
	return false
}

func validateImageMimeType(mimeType ImageMimeType) error {
	if !IsValidImageMimeType(mimeType) {
		return goerr.New("unsupported image format", goerr.V("mime_type", string(mimeType)))
	}
	return nil
}

const maxImageSize = 20 * 1024 * 1024 // 20MB

func validateImageSize(data []byte) error {
	if len(data) > maxImageSize {
		return goerr.New("image size exceeds maximum limit", goerr.V("size", len(data)), goerr.V("max_size", maxImageSize))
	}
	return nil
}

// ImageOption is a functional option for creating Image
type ImageOption func(*imageConfig) error

// imageConfig holds configuration for Image creation
type imageConfig struct {
	mimeType ImageMimeType
}

// WithMimeType explicitly sets the MIME type
func WithMimeType(mimeType ImageMimeType) ImageOption {
	return func(cfg *imageConfig) error {
		if err := validateImageMimeType(mimeType); err != nil {
			return err
		}
		cfg.mimeType = mimeType
		return nil
	}
}

// NewImage creates a new Image with automatic MIME type detection by default
func NewImage(data []byte, opts ...ImageOption) (Image, error) {
	cfg := &imageConfig{}

	// Apply options
	for _, opt := range opts {
		if err := opt(cfg); err != nil {
			return Image{}, err
		}
	}

	// Auto-detect MIME type if not specified
	if cfg.mimeType == "" {
		detectedType, err := detectImageMimeType(data)
		if err != nil {
			return Image{}, err
		}
		cfg.mimeType = detectedType
	}

	// Size validation
	if err := validateImageSize(data); err != nil {
		return Image{}, err
	}

	return Image{
		data:     data,
		mimeType: cfg.mimeType,
	}, nil
}

// NewImageFromReader creates a new Image from io.Reader
func NewImageFromReader(r io.Reader, opts ...ImageOption) (Image, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return Image{}, goerr.Wrap(err, "failed to read image data")
	}
	return NewImage(data, opts...)
}

// PDF represents a PDF document input for LLM
type PDF struct {
	data []byte
}

// isInput implements Input interface
func (p PDF) isInput() restrictedValue {
	return restrictedValue{}
}

// LogValue returns a slog.Value for the PDF
func (p PDF) LogValue() slog.Value {
	return slog.StringValue(p.String())
}

// String returns a string representation of the PDF
func (p PDF) String() string {
	return fmt.Sprintf("pdf (%d bytes)", len(p.data))
}

// Data returns the PDF data as bytes
func (p PDF) Data() []byte {
	return p.data
}

// Base64 returns the base64 encoded string of the PDF data
func (p PDF) Base64() string {
	return base64.StdEncoding.EncodeToString(p.data)
}

// MimeType returns the MIME type of the PDF
func (p PDF) MimeType() string {
	return "application/pdf"
}

// DefaultMaxPDFSize is the default maximum size for PDF data (32MB)
const DefaultMaxPDFSize = 32 * 1024 * 1024

// pdfMagicBytes is the magic bytes for PDF files
var pdfMagicBytes = []byte("%PDF-")

type pdfOption struct {
	maxSize int
}

// PDFOption is a functional option for PDF creation
type PDFOption func(*pdfOption)

// WithMaxPDFSize sets the maximum allowed size for PDF data
func WithMaxPDFSize(size int) PDFOption {
	return func(o *pdfOption) {
		o.maxSize = size
	}
}

func buildPDFOption(opts []PDFOption) pdfOption {
	o := pdfOption{maxSize: DefaultMaxPDFSize}
	for _, opt := range opts {
		opt(&o)
	}
	return o
}

// NewPDF creates a new PDF from byte data
func NewPDF(data []byte, opts ...PDFOption) (PDF, error) {
	o := buildPDFOption(opts)

	if len(data) == 0 {
		return PDF{}, goerr.New("PDF data is empty")
	}

	if len(data) > o.maxSize {
		return PDF{}, goerr.New("PDF size exceeds maximum limit", goerr.V("size", len(data)), goerr.V("max_size", o.maxSize))
	}

	if len(data) < len(pdfMagicBytes) || !bytes.Equal(data[:len(pdfMagicBytes)], pdfMagicBytes) {
		return PDF{}, goerr.New("invalid PDF format")
	}

	return PDF{data: data}, nil
}

// NewPDFFromReader creates a new PDF from io.Reader
func NewPDFFromReader(r io.Reader, opts ...PDFOption) (PDF, error) {
	o := buildPDFOption(opts)

	// Use LimitReader to prevent memory exhaustion from untrusted readers
	limitedReader := io.LimitReader(r, int64(o.maxSize)+1)
	data, err := io.ReadAll(limitedReader)
	if err != nil {
		return PDF{}, goerr.Wrap(err, "failed to read PDF data")
	}
	return NewPDF(data, opts...)
}
