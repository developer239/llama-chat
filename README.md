# LlamaCPP 🦙🦙

LlamaCPP is a C++ library designed for running language models using the [llama.cpp](https://github.com/your-org/llama.cpp) framework. It provides an easy-to-use interface for loading models, querying them, and streaming responses in C++ applications.

**Supported Systems:**

- MacOS
- Windows
- Linux

## What Can You Do With It?

- Load language models and run queries with advanced sampling options.
- Stream query responses in real-time.
- Encode and decode text to and from tokens.
- Easy integration with existing C++ CMake projects.

## Installation

### Add LlamaCPP as a Submodule

First, add this library as a submodule in your project:

```bash
$ git submodule add https://github.com/developer239/llama-wrapped-cmake externals/llama-cpp
```

Load the module's dependencies:

```bash
$ git submodule update --init --recursive
```

### Update Your CMake

In your project's `CMakeLists.txt`, add the following lines to include and link the LlamaCPP library:

```cmake
add_subdirectory(externals/llama-cpp)
target_link_libraries(<your_target> PRIVATE LlamaCPP)
```

## Usage

### Basic Usage

To use the LlamaCPP library, include the header and create an instance of the `LlamaWrapper` class. You can initialize the model and context separately, then run queries or stream responses with custom sampling parameters.

```cpp
#include "llama-wrapper.h"
#include <iostream>

int main() {
    LlamaWrapper llama;
    
    ModelParams model_params;
    model_params.n_gpu_layers = 1;  // Use 1 GPU layer
    
    if (!llama.InitializeModel("path/to/model", model_params)) {
        std::cerr << "Failed to initialize the model." << std::endl;
        return 1;
    }
    
    ContextParams ctx_params;
    ctx_params.n_ctx = 2048;  // Set context size to 2048
    
    if (!llama.InitializeContext(ctx_params)) {
        std::cerr << "Failed to initialize the context." << std::endl;
        return 1;
    }

    SamplingParams sampling_params;
    sampling_params.temperature = 0.7f;
    sampling_params.top_k = 50;
    sampling_params.top_p = 0.9f;
    sampling_params.max_tokens = 1000;

    std::string response = llama.RunQuery("How do I write hello world in C++?", sampling_params);
    std::cout << "Response: " << response << std::endl;

    return 0;
}
```

### Streaming Responses

You can also stream responses by providing a callback function. This is useful for real-time applications or long outputs.

```cpp
#include "llama-wrapper.h"
#include <iostream>

int main() {
    LlamaWrapper llama;
    
    // Initialize model and context as in the previous example

    SamplingParams sampling_params;
    sampling_params.temperature = 0.8f;
    sampling_params.max_tokens = 1000;

    llama.RunQueryStream("Tell me a story.", sampling_params, [](const std::string& piece) {
        std::cout << piece << std::flush;
    });

    return 0;
}
```

### Working with Tokens

You can encode and decode text to and from tokens:

```cpp
std::vector<llama_token> tokens = llama.Encode("Hello, world!", true);
std::string decoded = llama.Decode(tokens);
```

## API Reference

### LlamaWrapper Class

The `LlamaWrapper` class provides methods to interact with language models loaded through llama.cpp.

#### Public Methods

- `LlamaWrapper()`: Constructor. Initializes the LlamaWrapper object.
- `~LlamaWrapper()`: Destructor. Cleans up resources.
- `bool InitializeModel(const std::string& model_path, const ModelParams& params)`: Initializes the model with the specified path and parameters.
- `bool InitializeContext(const ContextParams& params)`: Initializes the context with the specified parameters.
- `std::string RunQuery(const std::string& prompt, const SamplingParams& params, bool add_bos = true)`: Runs a query with the given prompt and sampling parameters, and returns the result as a string.
- `void RunQueryStream(const std::string& prompt, const SamplingParams& params, const std::function<void(const std::string&)>& callback, bool add_bos = true)`: Streams the response to the given prompt, invoking the callback function with each piece of the response.
- `std::vector<llama_token> Encode(const std::string& text, bool add_bos = true)`: Encodes the given text into tokens.
- `std::string Decode(const std::vector<llama_token>& tokens)`: Decodes the given tokens into text.
- `llama_token TokenBos()`: Returns the Beginning of Sentence token.
- `llama_token TokenEos()`: Returns the End of Sentence token.
- `llama_token TokenNl()`: Returns the Newline token.

#### Structs

- `ModelParams`: Parameters for model initialization.
    - `n_gpu_layers` (int): Number of layers to offload to GPU. Set to 0 for CPU-only.
    - `vocab_only` (bool): Only load the vocabulary, no weights. It's useful when you only need to perform tokenization (converting text to token IDs) but don't need to generate text or perform inference. This significantly reduces memory usage and loading time.
    - `use_mmap` (bool): Use memory mapping for faster loading.
    - `use_mlock` (bool): Force system to keep model in RAM.

- `ContextParams`: Parameters for context initialization.
    - `n_ctx` (size_t): Size of the context window (in tokens).
    - `n_threads` (int): Number of threads to use for computation.
    - `n_batch` (int): Number of tokens to process in parallel.
    - `logits_all` (bool): Return logits for all tokens in the context.
    - `embedding` (bool): Embed input text (not used for text generation).

- `SamplingParams`: Parameters for text generation sampling.
    - `max_tokens` (size_t): Maximum number of tokens to generate.
    - `temperature` (float): Controls randomness in generation. Lower values make the model more deterministic.
    - `top_k` (int32_t): Limits sampling to the k most likely tokens. Lower values (e.g., 10-50) tend to produce more focused and deterministic outputs. Higher values (e.g., 100-1000) allow for more diverse and potentially creative outputs, but may also introduce more errors or irrelevant content. Setting top_k to a very high value (or the vocabulary size) effectively disables top-k sampling.
    - `top_p` (float): Limits sampling to a cumulative probability. Lower values (e.g., 0.1-0.3) make the output more focused and conservative. Higher values (e.g., 0.7-0.9) allow for more diversity but may lead to less coherent text. Setting top_p to 1.0 effectively disables top-p sampling.
    - `repeat_penalty` (float): Penalty for repeating tokens.
    - `frequency_penalty` (float): Penalty based on token frequency in the generated text.
    - `presence_penalty` (float): Penalty for tokens already present in the generated text.
    - `repeat_penalty_tokens` (std::vector<llama_token>): Tokens to consider for repeat penalty.

These structs allow fine-grained control over model initialization, context setup, and text generation. Adjust these parameters to optimize performance and output quality for your specific use case.
