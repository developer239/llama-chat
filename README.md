# LlamaChat 🦙🦙🦙

LlamaChat is a C++ library designed for running language models using the [llama.cpp](https://github.com/your-org/llama.cpp) framework. It provides an easy-to-use interface for loading models, querying them, and streaming responses in C++ applications.

**Supported Systems:**

- MacOS
- Windows
- Linux

## Installation

### Add LlamaChat as a Submodule

First, add this library as a submodule in your project:

```bash
$ git submodule add https://github.com/developer239/llama-wrapped-cmake externals/llama-chat
```

Load the module's dependencies:

```bash
$ git submodule update --init --recursive
```

### Update Your CMake

In your project's `CMakeLists.txt`, add the following lines to include and link the LlamaChat library:

```cmake
add_subdirectory(externals/llama-chat)
target_link_libraries(<your_target> PRIVATE LlamaChat)
```

## Usage

### Basic Usage

To use the LlamaChat library, include the header and create an instance of the `LlamaChat` class. You can initialize the model and context separately, then run queries or stream responses.

```cpp
#include "llama-chat.h"
#include <iostream>

int main() {
    LlamaChat llama;
    
    ModelParams modelParams;
    modelParams.nGpuLayers = 32;  // Adjust based on your GPU capabilities
    
    if (!llama.InitializeModel("path/to/model", modelParams)) {
        std::cerr << "Failed to initialize the model." << std::endl;
        return 1;
    }
    
    ContextParams ctxParams;
    ctxParams.nContext = 2048;
    
    if (!llama.InitializeContext(ctxParams)) {
        std::cerr << "Failed to initialize the context." << std::endl;
        return 1;
    }

    std::string systemPrompt = "You are a helpful AI assistant.";
    llama.SetSystemPrompt(systemPrompt);

    std::string userMessage = "How do I write hello world in C++?";

    llama.Prompt(userMessage, [](const std::string& piece) {
        std::cout << piece << std::flush;
    });

    return 0;
}
```

### Streaming Responses

The `Prompt` method implements streaming responses by providing a callback function. This is useful for long outputs.

## API Reference

### LlamaChat Class

The `LlamaChat` class provides methods to interact with language models loaded through llama.cpp.

#### Public Methods

- `LlamaChat()`: Constructor. Initializes the LlamaChat object.
- `~LlamaChat()`: Destructor. Cleans up resources.
- `bool InitializeModel(const std::string& modelPath, const ModelParams& params)`: Initializes the model with the specified path and parameters.
- `bool InitializeContext(const ContextParams& params)`: Initializes the context with the specified parameters.
- `void SetSystemPrompt(const std::string& systemPrompt)`: Sets the system prompt for the conversation.
- `void ResetConversation()`: Resets the conversation history.
- `void Prompt(const std::string& userMessage, const std::function<void(const std::string&)>& callback)`: Processes the user message and streams the response, invoking the callback function with each piece of the response.

#### Structs

- `LlamaToken`: Represents a token in the model's vocabulary.
    - `tokenId` (int): The unique identifier of the token.

- `ModelParams`: Parameters for model initialization.
    - `nGpuLayers` (int): Number of layers to offload to GPU. Set to 0 for CPU-only.
    - `vocabularyOnly` (bool): Only load the vocabulary, no weights.
    - `useMemoryMapping` (bool): Use memory mapping for faster loading.
    - `useModelLock` (bool): Force system to keep model in RAM.

- `ContextParams`: Parameters for context initialization.
    - `nContext` (size_t): Size of the context window (in tokens).
    - `nThreads` (int): Number of threads to use for computation.
    - `nBatch` (int): Number of tokens to process in parallel.

- `SamplingParams`: Parameters for text generation sampling.
    - `maxTokens` (size_t): Maximum number of tokens to generate.
    - `temperature` (float): Controls randomness in generation.
    - `topK` (int32_t): Limits sampling to the k most likely tokens.
    - `topP` (float): Limits sampling to a cumulative probability.
    - `repeatPenalty` (float): Penalty for repeating tokens.
    - `frequencyPenalty` (float): Penalty based on token frequency in generated text.
    - `presencePenalty` (float): Penalty for tokens already present in generated text.
    - `repeatPenaltyTokens` (std::vector<LlamaToken>): Tokens to consider for repeat penalty.
