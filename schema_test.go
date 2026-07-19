package gollem_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"regexp"
	"strings"
	"testing"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/llm/claude"
	"github.com/gollem-dev/gollem/llm/gemini"
	"github.com/gollem-dev/gollem/llm/openai"
	"github.com/m-mizutani/gt"
)

func TestToSchemaBasicTypes(t *testing.T) {
	type TestStruct struct {
		Name   string   `json:"name"`
		Age    int      `json:"age"`
		Score  float64  `json:"score"`
		Active bool     `json:"active"`
		Tags   []string `json:"tags"`
	}

	param, err := gollem.ToSchema(TestStruct{})
	gt.NoError(t, err)
	gt.Equal(t, param.Type, gollem.TypeObject)
	gt.Equal(t, len(param.Properties), 5)

	// Check string field
	gt.Equal(t, param.Properties["name"].Type, gollem.TypeString)

	// Check int field
	gt.Equal(t, param.Properties["age"].Type, gollem.TypeInteger)

	// Check float field
	gt.Equal(t, param.Properties["score"].Type, gollem.TypeNumber)

	// Check bool field
	gt.Equal(t, param.Properties["active"].Type, gollem.TypeBoolean)

	// Check array field
	gt.Equal(t, param.Properties["tags"].Type, gollem.TypeArray)
	gt.Equal(t, param.Properties["tags"].Items.Type, gollem.TypeString)
}

func TestToSchemaWithDescription(t *testing.T) {
	type User struct {
		Name string `json:"name" description:"User's full name"`
		Age  int    `json:"age" description:"Age in years"`
	}

	param, err := gollem.ToSchema(User{})
	gt.NoError(t, err)
	gt.Equal(t, param.Properties["name"].Description, "User's full name")
	gt.Equal(t, param.Properties["age"].Description, "Age in years")
}

func TestToSchemaWithEnum(t *testing.T) {
	type User struct {
		Role   string `json:"role" enum:"admin,user,guest"`
		Status string `json:"status" enum:"active, inactive, pending"`
	}

	param, err := gollem.ToSchema(User{})
	gt.NoError(t, err)
	gt.Equal(t, param.Properties["role"].Enum, []string{"admin", "user", "guest"})
	// Enum values should be trimmed
	gt.Equal(t, param.Properties["status"].Enum, []string{"active", "inactive", "pending"})
}

func TestToSchemaWithNumericConstraints(t *testing.T) {
	type Product struct {
		Price    float64 `json:"price" min:"0" max:"10000"`
		Quantity int     `json:"quantity" min:"1" max:"100"`
	}

	param, err := gollem.ToSchema(Product{})
	gt.NoError(t, err)

	// Check price constraints
	gt.NotNil(t, param.Properties["price"].Minimum)
	gt.V(t, *param.Properties["price"].Minimum).Equal(0.0)
	gt.NotNil(t, param.Properties["price"].Maximum)
	gt.V(t, *param.Properties["price"].Maximum).Equal(10000.0)

	// Check quantity constraints
	gt.NotNil(t, param.Properties["quantity"].Minimum)
	gt.V(t, *param.Properties["quantity"].Minimum).Equal(1.0)
	gt.NotNil(t, param.Properties["quantity"].Maximum)
	gt.V(t, *param.Properties["quantity"].Maximum).Equal(100.0)
}

func TestToSchemaWithStringConstraints(t *testing.T) {
	type User struct {
		Username string `json:"username" minLength:"3" maxLength:"20" pattern:"^[a-z0-9]+$"`
		Email    string `json:"email" pattern:"^[a-z@.]+$"`
	}

	param, err := gollem.ToSchema(User{})
	gt.NoError(t, err)

	// Check username constraints
	gt.NotNil(t, param.Properties["username"].MinLength)
	gt.V(t, *param.Properties["username"].MinLength).Equal(3)
	gt.NotNil(t, param.Properties["username"].MaxLength)
	gt.V(t, *param.Properties["username"].MaxLength).Equal(20)
	gt.Equal(t, param.Properties["username"].Pattern, "^[a-z0-9]+$")

	// Check email pattern
	gt.Equal(t, param.Properties["email"].Pattern, "^[a-z@.]+$")
}

func TestToSchemaWithArrayConstraints(t *testing.T) {
	type Post struct {
		Tags []string `json:"tags" minItems:"1" maxItems:"10"`
	}

	param, err := gollem.ToSchema(Post{})
	gt.NoError(t, err)

	gt.NotNil(t, param.Properties["tags"].MinItems)
	gt.V(t, *param.Properties["tags"].MinItems).Equal(1)
	gt.NotNil(t, param.Properties["tags"].MaxItems)
	gt.V(t, *param.Properties["tags"].MaxItems).Equal(10)
}

func TestToSchemaWithRequired(t *testing.T) {
	type User struct {
		Name  string `json:"name" required:"true"`
		Email string `json:"email" required:"true"`
		Age   int    `json:"age"`
	}

	param, err := gollem.ToSchema(User{})
	gt.NoError(t, err)

	// Check required fields - each property has its own Required bool
	gt.True(t, param.Properties["name"].Required)
	gt.True(t, param.Properties["email"].Required)
	gt.False(t, param.Properties["age"].Required)
}

func TestToSchemaNestedStruct(t *testing.T) {
	type Address struct {
		Street  string `json:"street" required:"true"`
		City    string `json:"city" required:"true"`
		Country string `json:"country" required:"true"`
	}

	type User struct {
		Name    string  `json:"name" required:"true"`
		Address Address `json:"address" required:"true"`
	}

	param, err := gollem.ToSchema(User{})
	gt.NoError(t, err)
	gt.Equal(t, param.Type, gollem.TypeObject)

	// Check required on top-level properties
	gt.True(t, param.Properties["name"].Required)
	gt.True(t, param.Properties["address"].Required)

	// Check nested struct
	address := param.Properties["address"]
	gt.Equal(t, address.Type, gollem.TypeObject)
	gt.Equal(t, len(address.Properties), 3)

	// Check all nested required fields
	gt.True(t, address.Properties["street"].Required)
	gt.True(t, address.Properties["city"].Required)
	gt.True(t, address.Properties["country"].Required)
}

func TestToSchemaArrayOfStructs(t *testing.T) {
	type Item struct {
		Name  string  `json:"name" required:"true"`
		Price float64 `json:"price" min:"0"`
	}

	type Order struct {
		Items []Item `json:"items" minItems:"1"`
	}

	param, err := gollem.ToSchema(Order{})
	gt.NoError(t, err)

	items := param.Properties["items"]
	gt.Equal(t, items.Type, gollem.TypeArray)
	gt.NotNil(t, items.MinItems)
	gt.V(t, *items.MinItems).Equal(1)

	// Check array element type
	gt.Equal(t, items.Items.Type, gollem.TypeObject)
	gt.Equal(t, len(items.Items.Properties), 2)
	gt.True(t, items.Items.Properties["name"].Required)
	gt.False(t, items.Items.Properties["price"].Required)
}

func TestToSchemaIgnoredFields(t *testing.T) {
	type User struct {
		Name     string `json:"name"`
		Password string `json:"-"` // Should be ignored
	}

	param, err := gollem.ToSchema(User{})
	gt.NoError(t, err)
	gt.Equal(t, len(param.Properties), 1) // Only "name" should be included
	gt.NotNil(t, param.Properties["name"])
	gt.Nil(t, param.Properties["Password"])
}

func TestToSchemaPointerType(t *testing.T) {
	type User struct {
		Name *string `json:"name"`
	}

	param, err := gollem.ToSchema(User{})
	gt.NoError(t, err)
	gt.Equal(t, param.Properties["name"].Type, gollem.TypeString)
}

func TestToSchemaComplexExample(t *testing.T) {
	type SecurityAlert struct {
		Severity        string   `json:"severity" enum:"low,medium,high,critical" required:"true"`
		ThreatType      string   `json:"threat_type" required:"true"`
		AffectedSystems []string `json:"affected_systems" minItems:"1"`
		IPAddresses     []string `json:"ip_addresses"`
		Confidence      float64  `json:"confidence" min:"0" max:"1"`
	}

	param, err := gollem.ToSchema(SecurityAlert{})
	gt.NoError(t, err)
	gt.Equal(t, param.Type, gollem.TypeObject)
	// Check that severity and threat_type are required
	gt.True(t, param.Properties["severity"].Required)
	gt.True(t, param.Properties["threat_type"].Required)

	// Check enum field
	severity := param.Properties["severity"]
	gt.Equal(t, severity.Type, gollem.TypeString)
	gt.Equal(t, len(severity.Enum), 4)

	// Check array with constraints
	affected := param.Properties["affected_systems"]
	gt.Equal(t, affected.Type, gollem.TypeArray)
	gt.NotNil(t, affected.MinItems)

	// Check numeric constraints
	confidence := param.Properties["confidence"]
	gt.NotNil(t, confidence.Minimum)
	gt.V(t, *confidence.Minimum).Equal(0.0)
	gt.NotNil(t, confidence.Maximum)
	gt.V(t, *confidence.Maximum).Equal(1.0)
}

func TestToSchemaInvalidTags(t *testing.T) {
	type InvalidMin struct {
		Value int `json:"value" min:"invalid"`
	}

	_, err := gollem.ToSchema(InvalidMin{})
	gt.Error(t, err)
	gt.True(t, errors.Is(err, gollem.ErrInvalidTag))
}

func TestToSchemaCyclicReference(t *testing.T) {
	type Node struct {
		Value string `json:"value"`
		Next  *Node  `json:"next"`
	}

	_, err := gollem.ToSchema(Node{})
	gt.Error(t, err)
	gt.True(t, errors.Is(err, gollem.ErrCyclicReference))
}

func TestMustToSchemaPanic(t *testing.T) {
	type Invalid struct {
		Chan chan int `json:"chan"` // Unsupported type
	}

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("MustToSchema should panic on unsupported type")
		}
	}()

	gollem.MustToSchema(Invalid{})
}

func TestToSchemaWithTitleAndDescription(t *testing.T) {
	type UserProfile struct {
		Name  string `json:"name" required:"true"`
		Email string `json:"email" required:"true"`
	}

	schema, err := gollem.ToSchema(UserProfile{})
	gt.NoError(t, err)

	// Set Title and Description
	schema.Title = "UserProfile"
	schema.Description = "Structured user profile information"

	gt.Equal(t, schema.Title, "UserProfile")
	gt.Equal(t, schema.Description, "Structured user profile information")
	gt.Equal(t, schema.Type, gollem.TypeObject)

	// Check required on properties
	gt.True(t, schema.Properties["name"].Required)
	gt.True(t, schema.Properties["email"].Required)
}

func TestToSchemaMapType(t *testing.T) {
	type Config struct {
		Settings map[string]string `json:"settings"`
	}

	param, err := gollem.ToSchema(Config{})
	gt.NoError(t, err)
	gt.Equal(t, param.Properties["settings"].Type, gollem.TypeObject)
}

func TestToSchemaAllIntegerTypes(t *testing.T) {
	type Numbers struct {
		I8  int8    `json:"i8"`
		I16 int16   `json:"i16"`
		I32 int32   `json:"i32"`
		I64 int64   `json:"i64"`
		U   uint    `json:"u"`
		U8  uint8   `json:"u8"`
		U16 uint16  `json:"u16"`
		U32 uint32  `json:"u32"`
		U64 uint64  `json:"u64"`
		F32 float32 `json:"f32"`
	}

	param, err := gollem.ToSchema(Numbers{})
	gt.NoError(t, err)

	// All int types should be TypeInteger
	gt.Equal(t, param.Properties["i8"].Type, gollem.TypeInteger)
	gt.Equal(t, param.Properties["i16"].Type, gollem.TypeInteger)
	gt.Equal(t, param.Properties["i32"].Type, gollem.TypeInteger)
	gt.Equal(t, param.Properties["i64"].Type, gollem.TypeInteger)
	gt.Equal(t, param.Properties["u"].Type, gollem.TypeInteger)
	gt.Equal(t, param.Properties["u8"].Type, gollem.TypeInteger)
	gt.Equal(t, param.Properties["u16"].Type, gollem.TypeInteger)
	gt.Equal(t, param.Properties["u32"].Type, gollem.TypeInteger)
	gt.Equal(t, param.Properties["u64"].Type, gollem.TypeInteger)

	// float32 should be TypeNumber
	gt.Equal(t, param.Properties["f32"].Type, gollem.TypeNumber)
}

func TestToSchemaArrayType(t *testing.T) {
	type FixedArray struct {
		Items [5]string `json:"items"`
	}

	param, err := gollem.ToSchema(FixedArray{})
	gt.NoError(t, err)
	gt.Equal(t, param.Properties["items"].Type, gollem.TypeArray)
	gt.Equal(t, param.Properties["items"].Items.Type, gollem.TypeString)
}

func TestToSchemaUnexportedFields(t *testing.T) {
	type User struct {
		Name     string `json:"name"`
		password string //nolint:unused // intentionally unexported for testing
	}

	param, err := gollem.ToSchema(User{})
	gt.NoError(t, err)
	gt.Equal(t, len(param.Properties), 1) // Only exported field
	gt.NotNil(t, param.Properties["name"])
	gt.Nil(t, param.Properties["password"])
}

func TestToSchemaNilInput(t *testing.T) {
	_, err := gollem.ToSchema(nil)
	gt.Error(t, err)
	gt.True(t, errors.Is(err, gollem.ErrUnsupportedType))
}

func TestToSchemaInvalidRequiredTag(t *testing.T) {
	type User struct {
		Name string `json:"name" required:"invalid"`
	}

	_, err := gollem.ToSchema(User{})
	gt.Error(t, err)
	gt.True(t, errors.Is(err, gollem.ErrInvalidTag))
}

func TestToSchemaInvalidMaxTag(t *testing.T) {
	type Product struct {
		Price float64 `json:"price" max:"not_a_number"`
	}

	_, err := gollem.ToSchema(Product{})
	gt.Error(t, err)
	gt.True(t, errors.Is(err, gollem.ErrInvalidTag))
}

func TestToSchemaInvalidMinLengthTag(t *testing.T) {
	type User struct {
		Name string `json:"name" minLength:"abc"`
	}

	_, err := gollem.ToSchema(User{})
	gt.Error(t, err)
	gt.True(t, errors.Is(err, gollem.ErrInvalidTag))
}

func TestToSchemaInvalidMaxItemsTag(t *testing.T) {
	type Post struct {
		Tags []string `json:"tags" maxItems:"xyz"`
	}

	_, err := gollem.ToSchema(Post{})
	gt.Error(t, err)
	gt.True(t, errors.Is(err, gollem.ErrInvalidTag))
}

func TestToSchemaUnsupportedType(t *testing.T) {
	type Invalid struct {
		Ch chan int `json:"ch"`
	}

	_, err := gollem.ToSchema(Invalid{})
	gt.Error(t, err)
	gt.True(t, errors.Is(err, gollem.ErrUnsupportedType))
}

// Ptr returns a pointer to a value of any type
func Ptr[T any](v T) *T {
	return &v
}

// createComplexBookSchema creates a complex schema for integration testing
func createComplexBookSchema() *gollem.Parameter {
	return &gollem.Parameter{
		Title:       "BookReview",
		Description: "A detailed book review with metadata",
		Type:        gollem.TypeObject,
		Properties: map[string]*gollem.Parameter{
			"book": {
				Type:     gollem.TypeObject,
				Required: true,
				Properties: map[string]*gollem.Parameter{
					"title": {
						Type:        gollem.TypeString,
						Description: "Book title",
						Required:    true,
					},
					"author": {
						Type:        gollem.TypeString,
						Description: "Author name",
						Required:    true,
					},
					"publishedYear": {
						Type:        gollem.TypeInteger,
						Description: "Year of publication",
						Minimum:     Ptr(1000.0),
						Maximum:     Ptr(2100.0),
					},
					"isbn": {
						Type:        gollem.TypeString,
						Description: "ISBN number",
						Pattern:     "^[0-9-]+$",
					},
				},
				Description: "Book information",
			},
			"review": {
				Type:     gollem.TypeObject,
				Required: true,
				Properties: map[string]*gollem.Parameter{
					"rating": {
						Type:        gollem.TypeNumber,
						Description: "Rating from 1.0 to 5.0",
						Minimum:     Ptr(1.0),
						Maximum:     Ptr(5.0),
						Required:    true,
					},
					"summary": {
						Type:        gollem.TypeString,
						Description: "Brief review summary",
						MinLength:   Ptr(10),
						MaxLength:   Ptr(500),
						Required:    true,
					},
					"pros": {
						Type: gollem.TypeArray,
						Items: &gollem.Parameter{
							Type: gollem.TypeString,
						},
						Description: "Positive aspects",
						MinItems:    Ptr(1),
					},
					"cons": {
						Type: gollem.TypeArray,
						Items: &gollem.Parameter{
							Type: gollem.TypeString,
						},
						Description: "Negative aspects",
					},
				},
				Description: "Review content",
			},
			"tags": {
				Type: gollem.TypeArray,
				Items: &gollem.Parameter{
					Type: gollem.TypeString,
					Enum: []string{"fiction", "non-fiction", "mystery", "romance", "sci-fi", "fantasy", "biography", "history", "technical"},
				},
				Description: "Book genre tags",
				MinItems:    Ptr(1),
				MaxItems:    Ptr(5),
				Required:    true,
			},
			"recommended": {
				Type:        gollem.TypeBoolean,
				Description: "Whether the book is recommended",
				Required:    true,
			},
		},
	}
}

// validateJSONAgainstSchema validates that the JSON response matches the expected schema
func validateJSONAgainstSchema(t *testing.T, jsonStr string, schema *gollem.Parameter) {
	t.Helper()

	// Parse JSON
	var data map[string]any
	err := json.Unmarshal([]byte(jsonStr), &data)
	if err != nil {
		t.Fatalf("response should be valid JSON: %v", err)
	}

	// Validate the root object
	validateParameter(t, "", data, schema)
}

// validateParameter recursively validates a value against a parameter schema
func validateParameter(t *testing.T, path string, value any, param *gollem.Parameter) {
	t.Helper()

	if path == "" {
		path = "root"
	}

	// Type validation
	switch param.Type {
	case gollem.TypeString:
		strVal, ok := value.(string)
		if !ok {
			t.Errorf("%s should be string, got %T", path, value)
			return
		}

		// Pattern validation
		if param.Pattern != "" {
			matched, err := regexp.MatchString(param.Pattern, strVal)
			if err != nil {
				t.Errorf("%s pattern validation failed: %v", path, err)
			} else if !matched {
				t.Errorf("%s value %q does not match pattern %q", path, strVal, param.Pattern)
			}
		}

		// Length constraints
		if param.MinLength != nil && len(strVal) < *param.MinLength {
			t.Errorf("%s length %d is less than minLength %d", path, len(strVal), *param.MinLength)
		}
		if param.MaxLength != nil && len(strVal) > *param.MaxLength {
			t.Errorf("%s length %d exceeds maxLength %d", path, len(strVal), *param.MaxLength)
		}

		// Enum validation
		if param.Enum != nil {
			found := false
			for _, enumVal := range param.Enum {
				if strVal == enumVal {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("%s value %q is not in enum %v", path, strVal, param.Enum)
			}
		}

	case gollem.TypeInteger:
		numVal, ok := value.(float64)
		if !ok {
			t.Errorf("%s should be number, got %T", path, value)
			return
		}

		// Check if it's actually an integer
		if numVal != float64(int(numVal)) {
			t.Errorf("%s should be integer, got float %f", path, numVal)
		}

		// Range constraints
		if param.Minimum != nil && numVal < *param.Minimum {
			t.Errorf("%s value %f is less than minimum %f", path, numVal, *param.Minimum)
		}
		if param.Maximum != nil && numVal > *param.Maximum {
			t.Errorf("%s value %f exceeds maximum %f", path, numVal, *param.Maximum)
		}

	case gollem.TypeNumber:
		numVal, ok := value.(float64)
		if !ok {
			t.Errorf("%s should be number, got %T", path, value)
			return
		}

		// Range constraints
		if param.Minimum != nil && numVal < *param.Minimum {
			t.Errorf("%s value %f is less than minimum %f", path, numVal, *param.Minimum)
		}
		if param.Maximum != nil && numVal > *param.Maximum {
			t.Errorf("%s value %f exceeds maximum %f", path, numVal, *param.Maximum)
		}

	case gollem.TypeBoolean:
		if _, ok := value.(bool); !ok {
			t.Errorf("%s should be boolean, got %T", path, value)
		}

	case gollem.TypeArray:
		arrVal, ok := value.([]any)
		if !ok {
			t.Errorf("%s should be array, got %T", path, value)
			return
		}

		// Array length constraints
		if param.MinItems != nil && len(arrVal) < *param.MinItems {
			t.Errorf("%s array length %d is less than minItems %d", path, len(arrVal), *param.MinItems)
		}
		if param.MaxItems != nil && len(arrVal) > *param.MaxItems {
			t.Errorf("%s array length %d exceeds maxItems %d", path, len(arrVal), *param.MaxItems)
		}

		// Validate each item
		if param.Items != nil {
			for i, item := range arrVal {
				itemPath := fmt.Sprintf("%s[%d]", path, i)
				validateParameter(t, itemPath, item, param.Items)
			}
		}

	case gollem.TypeObject:
		objVal, ok := value.(map[string]any)
		if !ok {
			t.Errorf("%s should be object, got %T", path, value)
			return
		}

		// Check required fields by checking each property's Required flag
		if param.Properties != nil {
			for propName, propSchema := range param.Properties {
				if propSchema.Required {
					if _, exists := objVal[propName]; !exists {
						t.Errorf("%s missing required field %q", path, propName)
					}
				}
			}

			// Validate each property
			for propName, propSchema := range param.Properties {
				propValue, exists := objVal[propName]
				if !exists {
					// Skip optional fields
					continue
				}
				propPath := fmt.Sprintf("%s.%s", path, propName)
				validateParameter(t, propPath, propValue, propSchema)
			}
		}
	}
}

// TestSchemaIntegration tests JSON schema functionality with real LLM clients
func TestSchemaIntegration(t *testing.T) {
	t.Parallel()

	testFn := func(t *testing.T, newClient func(t *testing.T) (gollem.LLMClient, error)) {
		ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
		defer cancel()

		client, err := newClient(t)
		gt.NoError(t, err)

		// Create complex test schema
		schema := createComplexBookSchema()

		// Create session with JSON content type and response schema
		session, err := client.NewSession(ctx,
			gollem.WithSessionContentType(gollem.ContentTypeJSON),
			gollem.WithSessionResponseSchema(schema),
		)
		gt.NoError(t, err)

		// Generate content with a complex prompt
		prompt := `Create a book review for "1984" by George Orwell.
		Published in 1949, ISBN 978-0451524935.
		Give it a 4.5 rating.
		Summary: A dystopian masterpiece exploring surveillance and totalitarianism.
		Pros: Thought-provoking, relevant today, excellent world-building.
		Cons: Dark and depressing at times.
		Tags: fiction, sci-fi.
		Highly recommended.`

		resp, err := session.Generate(ctx, []gollem.Input{gollem.Text(prompt)}, gollem.WithMaxTokens(maxTestTokens))
		gt.NoError(t, err)
		gt.NotNil(t, resp)
		gt.True(t, len(resp.Texts) > 0)

		// Get the JSON response
		jsonResponse := strings.Join(resp.Texts, "")
		t.Logf("JSON Response: %s", jsonResponse)

		// Validate the response matches the schema
		validateJSONAgainstSchema(t, jsonResponse, schema)

		// Parse and validate specific fields
		var bookReview map[string]any
		err = json.Unmarshal([]byte(jsonResponse), &bookReview)
		gt.NoError(t, err)

		// Validate book object exists and has required fields
		if bookObj, ok := bookReview["book"].(map[string]any); ok {
			if title, ok := bookObj["title"].(string); ok {
				if !strings.Contains(strings.ToLower(title), "1984") {
					t.Logf("Warning: title should contain '1984', got: %s", title)
				}
			} else {
				t.Error("book.title should be a string")
			}

			if author, ok := bookObj["author"].(string); ok {
				if !strings.Contains(strings.ToLower(author), "orwell") {
					t.Logf("Warning: author should contain 'Orwell', got: %s", author)
				}
			} else {
				t.Error("book.author should be a string")
			}
		} else {
			t.Error("book object should exist")
		}

		// Validate review object
		if reviewObj, ok := bookReview["review"].(map[string]any); ok {
			if rating, ok := reviewObj["rating"].(float64); ok {
				if rating < 1.0 || rating > 5.0 {
					t.Errorf("rating should be between 1.0 and 5.0, got: %f", rating)
				}
			} else {
				t.Error("review.rating should be a number")
			}

			if summary, ok := reviewObj["summary"].(string); ok {
				if len(summary) < 10 {
					t.Errorf("summary should be at least 10 characters, got: %d", len(summary))
				}
			} else {
				t.Error("review.summary should be a string")
			}

			// Validate pros array
			if pros, ok := reviewObj["pros"].([]any); ok {
				if len(pros) < 1 {
					t.Error("pros should have at least 1 item")
				}
			}
		} else {
			t.Error("review object should exist")
		}

		// Validate tags array
		if tags, ok := bookReview["tags"].([]any); ok {
			if len(tags) < 1 || len(tags) > 5 {
				t.Errorf("tags should have 1-5 items, got: %d", len(tags))
			}
		} else {
			t.Error("tags should be an array")
		}

		// Validate recommended boolean
		if _, ok := bookReview["recommended"].(bool); !ok {
			t.Error("recommended should be a boolean")
		}
	}

	t.Run("OpenAI", func(t *testing.T) {
		t.Parallel()
		apiKey, ok := os.LookupEnv("TEST_OPENAI_API_KEY")
		if !ok {
			t.Skip("TEST_OPENAI_API_KEY is not set")
		}
		testFn(t, func(t *testing.T) (gollem.LLMClient, error) {
			return openai.New(context.Background(), apiKey)
		})
	})

	t.Run("Claude", func(t *testing.T) {
		apiKey, ok := os.LookupEnv("TEST_CLAUDE_API_KEY")
		if !ok {
			t.Skip("TEST_CLAUDE_API_KEY is not set")
		}
		testFn(t, func(t *testing.T) (gollem.LLMClient, error) {
			return claude.New(context.Background(), apiKey)
		})
	})

	t.Run("Gemini", func(t *testing.T) {
		t.Parallel()
		projectID, ok := os.LookupEnv("TEST_GCP_PROJECT_ID")
		if !ok {
			t.Skip("TEST_GCP_PROJECT_ID is not set")
		}
		location, ok := os.LookupEnv("TEST_GCP_LOCATION")
		if !ok {
			t.Skip("TEST_GCP_LOCATION is not set")
		}
		var opts []gemini.Option
		if model := os.Getenv("TEST_GCP_MODEL"); model != "" {
			opts = append(opts, gemini.WithModel(model))
		}
		testFn(t, func(t *testing.T) (gollem.LLMClient, error) {
			return gemini.New(context.Background(), projectID, location, opts...)
		})
	})
}
