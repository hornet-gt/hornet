

struct cmp {
    edge_t a;
    edge_t b;
    __host__ __device__ __forceinline__
    cmp() {};

    __host__ __device__ __forceinline__
    cmp(edge_t _a, edge_t _b) {
        printf("(%d,%d)\n", _a, _b);
        a = _a;
        b = _b;
    }
    __host__ __device__ __forceinline__
    bool operator==(const cmp& rhs) const {
        printf("(%d,%d) == (%d,%d)\n", a, b, rhs.a, rhs.b);
        return a == rhs.a && b == rhs.b;
    }
};

class Iterator : public std::iterator<std::input_iterator_tag, cmp>
{
public:
    using difference_type = typename std::iterator<std::random_access_iterator_tag, edge_t>::difference_type;

    //Iterator() : _ptr1(nullptr), _ptr2(nullptr) {}
    __host__ __device__ __forceinline__
    Iterator(edge_t* ptr1, edge_t* ptr2) : _ptr1(ptr1), _ptr2(ptr2) {}
    //Iterator(const Iterator &rhs) : _ptr(rhs._ptr) {}
    /* inline Iterator& operator=(Type* rhs) {_ptr = rhs; return *this;} */
    /* inline Iterator& operator=(const Iterator &rhs) {_ptr = rhs._ptr; return *this;} */
    __host__ __device__ __forceinline__
    Iterator& operator+=(difference_type rhs) {
        _ptr1 += rhs;
        _ptr2 += rhs;
        return *this;
    }
    //inline Iterator& operator-=(difference_type rhs) {_ptr -= rhs; return *this;}
    //__host__ __device__ __forceinline__
    //edge_t& operator*() const {return *_ptr1;}

    //__host__ __device__ __forceinline__
    //edge_t* operator->() const {return _ptr1;}

    __host__ __device__ __forceinline__
    cmp operator[](difference_type rhs) const {
        return cmp(_ptr1[rhs], _ptr2[rhs]);
    }

    //__host__ __device__ __forceinline__
    //Iterator& operator++() { ++_ptr1; ++_ptr2; return *this; }
    //inline Iterator& operator--() {--_ptr; return *this;}
    //__host__ __device__ __forceinline__
    //Iterator operator++(int) {Iterator tmp(*this); ++_ptr1;
    //                                       ++_ptr2; return tmp;}

    __host__ __device__ __forceinline__
    Iterator operator+(difference_type rhs) const { return Iterator(_ptr1+rhs, _ptr2+rhs);}

    //inline Iterator operator--(int) const {Iterator tmp(*this); --_ptr; return tmp;}
    /*inline Iterator operator+(const Iterator& rhs) {
        return Iterator(_ptr1 + rhs._ptr1, _ptr2 + rhs._ptr2);
    }*/
    /*inline difference_type operator-(const Iterator& rhs) const {return Iterator(_ptr-rhs.ptr);}
    inline Iterator operator-(difference_type rhs) const {return Iterator(_ptr-rhs);}
    friend inline Iterator operator+(difference_type lhs, const Iterator& rhs) {return Iterator(lhs+rhs._ptr);}
    friend inline Iterator operator-(difference_type lhs, const Iterator& rhs) {return Iterator(lhs-rhs._ptr);}
*/
    __host__ __device__ __forceinline__
    bool operator==(const Iterator& rhs) const { return _ptr1 == rhs._ptr1;}

    //__host__ __device__ __forceinline__
    //bool operator!=(const Iterator& rhs) const { return _ptr1 != rhs._ptr1; }
    /*inline bool operator>(const Iterator& rhs) const {return _ptr > rhs._ptr;}
    inline bool operator<(const Iterator& rhs) const {return _ptr < rhs._ptr;}
    inline bool operator>=(const Iterator& rhs) const {return _ptr >= rhs._ptr;}
    inline bool operator<=(const Iterator& rhs) const {return _ptr <= rhs._ptr;}
*/
private:
    edge_t* _ptr1;
    edge_t* _ptr2;
};
