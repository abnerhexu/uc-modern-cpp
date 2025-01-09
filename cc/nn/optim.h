namespace optim {

class Optimizer {
public:
    Optimizer();
    virtual void step() = 0;
};

class SGDOptimizer: public Optimizer {
public:
    SGDOptimizer();
    void step();
};

}