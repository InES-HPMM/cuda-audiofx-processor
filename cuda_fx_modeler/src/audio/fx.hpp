

class Fx {
   public:
    Fx() {};
    virtual ~Fx() {};
    virtual void configure(size_t buf_size, size_t n_channels) = 0;
    virtual void setup() = 0;
    virtual void process(float** src, float** dest) = 0;
};
