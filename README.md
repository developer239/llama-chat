# LlamaCPP 🦙🦙

LlamaCPP is a C++ library designed for running language models using the [llama.cpp](https://github.com/your-org/llama.cpp) framework. It provides an easy-to-use interface for loading models, querying them, and streaming responses in C++ applications.

**Supported Systems:**

- MacOS
- Windows
- Linux

## What Can You Do With It?

- Load language models and run queries.
- Stream query responses in real-time.
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

To use the LlamaCPP library, include the header and use the singleton instance of the `LlamaWrapper` class. You can initialize it with the path to your model and a context size, then run queries or stream responses.

```cpp
#include "llama-wrapper.h"
#include <iostream>

int main() {
    auto& llama = LlamaWrapper::Instance();
    if (!llama.Initialize("path/to/model", 80000)) {
        std::cerr << "Failed to initialize the model." << std::endl;
        return 1;
    }

    std::string response = llama.RunQuery("How do I write hello world in C++?", 1000);
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
    auto& llama = LlamaWrapper::Instance();
    if (!llama.Initialize("path/to/model", 80000)) {
        std::cerr << "Failed to initialize the model." << std::endl;
        return 1;
    }

    llama.RunQueryStream("Tell me a story.", 1000, [](const std::string& piece) {
        std::cout << piece << std::flush;
    });

    return 0;
}
```

## API Reference

### LlamaWrapper Class

The `LlamaWrapper` class provides methods to interact with language models loaded through llama.cpp. It uses a singleton pattern to ensure only one instance interacts with the underlying resources.

#### Public Methods

- `static LlamaWrapper& Instance()`: Returns the singleton instance of the `LlamaWrapper` class.
- `bool Initialize(const std::string& model_path, size_t context_size = 80000)`: Initializes the model with the specified path and context size.
- `std::string RunQuery(const std::string& prompt, size_t max_tokens = 1000)`: Runs a query with the given prompt and returns the result as a string.
- `void RunQueryStream(const std::string& prompt, size_t max_tokens, const std::function<void(const std::string&)>& callback)`: Streams the response to the given prompt, invoking the callback function with each piece of the response.
