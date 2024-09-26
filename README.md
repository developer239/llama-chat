# LlamaCPP 🦙🦙 [update]

LlamaCPP is a C++ library designed for running language models using
the [llama.cpp](https://github.com/your-org/llama.cpp) framework. It provides an easy-to-use interface for loading
models, querying them, and streaming responses in C++ applications.

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

To use the LlamaCPP library, include the header and create an instance of the `LlamaWrapper` class. You can initialize
the model and context separately, then run queries or stream responses with custom sampling parameters.

```cpp
#include "llama-wrapper.h"
#include <iostream>

int main() {
    LlamaWrapper llama;
    
    ModelParams model_params;
    model_params.nGpuLayers = 1;  // M1 MacPro has 32 GPU Cores (run: `system_profiler SPDisplaysDataType`)
    
    if (!llama.InitializeModel("path/to/model", model_params)) {
        std::cerr << "Failed to initialize the model." << std::endl;
        return 1;
    }
    
    ContextParams ctx_params;
    ctx_params.nContext = 2048;  // Maximum is 128k
    
    if (!llama.InitializeContext(ctx_params)) {
        std::cerr << "Failed to initialize the context." << std::endl;
        return 1;
    }

    // see Structs section to learn about these parameters
    SamplingParams sampling_params;
    sampling_params.temperature = 0.7f;
    sampling_params.topK = 50;
    sampling_params.topP = 0.9f;
    sampling_params.maxTokens = 1000;

    std::string systemPrompt = "You are a helpful AI assistant.";
    std::string userMessage = "How do I write hello world in C++?";

    llama.RunQueryStream(systemPrompt, userMessage, sampling_params, [](const std::string& piece) {
        std::cout << piece << std::flush;
    });

    return 0;
}
```

### Streaming Responses

The `RunQueryStream` method already implements streaming responses by providing a callback function. This is useful for
real-time applications or long outputs.

### Working with Tokens

You can encode and decode text to and from tokens:

```cpp
std::vector<LlamaToken> tokens = llama.Encode("Hello, world!", true);
std::string decoded = llama.Decode(tokens);
```

## API Reference

### LlamaWrapper Class

The `LlamaWrapper` class provides methods to interact with language models loaded through llama.cpp.

#### Public Methods

- `LlamaWrapper()`: Constructor. Initializes the LlamaWrapper object.
- `~LlamaWrapper()`: Destructor. Cleans up resources.
- `bool InitializeModel(const std::string& modelPath, const ModelParams& params)`: Initializes the model with the
  specified path and parameters.
- `bool InitializeContext(const ContextParams& params)`: Initializes the context with the specified parameters.
- `void RunQueryStream(const std::string& systemPrompt, const std::string& userMessage, const SamplingParams& params, const std::function<void(const std::string&)>& callback)`:
  Streams the response to the given system prompt and user message, invoking the callback function with each piece of
  the response.
- `std::vector<LlamaToken> Encode(const std::string& text, bool addBos = true)`: Encodes the given text into tokens.
- `std::string Decode(const std::vector<LlamaToken>& tokens)`: Decodes the given tokens into text.
- `LlamaToken TokenBos()`: Returns the Beginning of Sentence token.
- `LlamaToken TokenEos()`: Returns the End of Sentence token.
- `LlamaToken TokenNl()`: Returns the Newline token.

#### Structs

- `LlamaToken`: Represents a token in the model's vocabulary.
    - `tokenId` (int): The unique identifier of the token.

- `ModelParams`: Parameters for model initialization.
    - `nGpuLayers` (int): Number of layers to offload to GPU. Set to 0 for CPU-only.
        - Default: 0 (CPU-only)
        - Higher values offload more layers to GPU, potentially increasing performance but requiring more GPU memory.
    - `vocabularyOnly` (bool): Only load the vocabulary, no weights.
        - Default: false
        - When true, it's useful for tokenization tasks, significantly reducing memory usage and loading time.
    - `useMemoryMapping` (bool): Use memory mapping for faster loading.
        - Default: true
        - Enables faster model loading but may use more virtual memory.
    - `useModelLock` (bool): Force system to keep model in RAM.
        - Default: false
        - When true, prevents the model from being swapped out, potentially improving performance but increasing memory
          pressure.

- `ContextParams`: Parameters for context initialization.
    - `nContext` (size_t): Size of the context window (in tokens).
        - Default: 4096
        - Larger values allow for longer context but require more memory.
    - `nThreads` (int): Number of threads to use for computation.
        - Default: 6
        - Higher values may improve performance on multi-core systems but can lead to diminishing returns.
    - `nBatch` (int): Number of tokens to process in parallel.
        - Default: 512
        - Larger values may improve performance but require more memory.

- `SamplingParams`: Parameters for text generation sampling.
    - `maxTokens` (size_t): Maximum number of tokens to generate.
        - Default: 1000
        - Higher values allow for longer generated text.
    - `temperature` (float): Controls randomness in generation.
        - Default: 0.8
        - Lower values (e.g., 0.2-0.5) make output more deterministic and focused.
        - Higher values (e.g., 1.0-1.5) increase randomness and creativity.
    - `topK` (int32_t): Limits sampling to the k most likely tokens.
        - Default: 45
        - Lower values (e.g., 10-50) produce more focused outputs.
        - Higher values (e.g., 100-1000) allow for more diversity.
        - Set to 0 or vocabulary size to disable.
    - `topP` (float): Limits sampling to a cumulative probability.
        - Default: 0.95
        - Lower values (e.g., 0.1-0.3) make output more focused and conservative.
        - Higher values (e.g., 0.7-0.9) allow for more diversity.
        - Set to 1.0 to disable.
    - `repeatPenalty` (float): Penalty for repeating tokens.
        - Default: 1.1
        - Higher values (e.g., 1.2-1.5) more strongly discourage repetition.
        - Set to 1.0 to disable.
    - `frequencyPenalty` (float): Penalty based on token frequency in generated text.
        - Default: 0.0 (disabled)
        - Positive values discourage frequent tokens, negative values encourage them.
    - `presencePenalty` (float): Penalty for tokens already present in generated text.
        - Default: 0.0 (disabled)
        - Positive values discourage tokens already present, negative values encourage them.
    - `repeatPenaltyTokens` (std::vector<LlamaToken>): Tokens to consider for repeat penalty.
        - Default: empty vector
        - Specifies which tokens to apply the repeat penalty to.

These structs allow fine-grained control over model initialization, context setup, and text generation. Adjust these
parameters to optimize performance and output quality for your specific use case.
