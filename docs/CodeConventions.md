# Code Convention #

extending Google/LLVM stype

* **Includes**
    - *First*: class header
    - *Second*: project includes (in alphabetical order) (`"name"` syntax)
    - *Third*: external includes (in alphabetical order) (`<name>` syntax)
    - Always prefer C++ includes over C includes (`<assert.h>` -> `<cassert>`)
    - DEVICE/HOST
        - Host Code, Host/Device code, CUDA API: `.hpp/.cpp`
        - Device code `.cuh/.cu` (minimize code compiled by nvcc)


* **General**
    - Always separate declaration `.hpp` from definition `.i.hpp/cpp`
    - Doxygen comment must be done only in `.hpp/.cuh` files
    - Take care of code alignment to make the code more readable
    - Use whether possible the `std` library
    - Do not use `Boost` library
    - Struct/Class variable members should be ordered by size


* **C++ Modernize**
    - `NULL` to `nullptr`
    - `typename` to `using`
    - `malloc/free` to `new[]/delete[]`
    - *old style cast* (int) to `reinterpret_cast`, `static_cast` and
      `const_cast`
    - `#pragma once` instead of `#ifdef` as include guard


* **Types**
    - `long long int/unsigned` to `int64_t/uint64_t`
    - `int32_t` to `int`
    - `size_t` to indicate sizes


* **Style and Formatting**
    - Tab '\t' is forbidden, use instead 4 spaces
    - Only linux new lines '\n' are allowed
    - 80 columns max for code and comments (terminal size)
    - *Functions*:
        - *Expensive functions*: Lower case start, camel style
            (ex. `expensiveFunction`)
        - *Cheap functions:*: lower case, underscore (`cheap_function`)
    - *Constant Value*:    upper case, underscore
    - *Macro*:             upper case, underscore
    - *Private Variables*: lower case, underscore (`_var_name`)
    - *Device variable*: `d_` prefix (`d_var`)
    - *Host variable*:   `h_` prefix (`h_var`) (use only to avoid confusion)
    - *Class name/Complex type*: upper case, camel style
    - *Simple type*:  lower case, underscore, `_t` postfix (`weight_t`)
    - *Namespace*: lower case, underscore (`my_namespace`)
    - *Namespace end*: close with `// namespace <name>`
    - *variable names*: not too short, not too long
    - *File name*: upper case, camel style
    - *Doxygen comment*: `@`
    - *Curly Bracket*: inline (`for (...) {`)
    - *Conditional Statement* (`if, for, while`): no bracket for one line body
    - *Separate words from symbols*: `if (`, `var = x + y;`


* **Safe Code**
    - `explicit` keyword costructors
    - `noexcept` keyword for all methods that do not perform file IO
    - method declaration `const` whenever possible
    - use `struct` only for passive structure, `class` otherwise.
    - All class members must be `private`, except `static` constants
    - Initialize *all* class members in class declaration `{ 0 }` if they do not
      depends on costructor, in costructor initializer list otherwise
    - Global namespaces `using namespace std` are forbidden in *all* project
      files. They are allowed *only* in test programs after includes.
    - *Function parameters*:
        - Inputs, then outputs
        - Prefer whether possible reference instead pointer
        - Do not use `const` for parameters passed by value
    - *Class/Struct*: first variable members, then methods.
                      First public members, then protected, then private.
