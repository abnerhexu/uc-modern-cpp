#include "../checker.h"

// READ: 派生类 <https://zh.cppreference.com/w/cpp/language/derived_class>

class Instruction {
public:
    uint32_t raw;
public:
    Instruction(uint32_t raw) : raw(raw) {}
};

class RTypeInstruction: public Instruction {
public:
    RTypeInstruction(uint32_t raw) : Instruction(raw) {}
    void get_rs1() {
        std::cout << ((this->raw >> 15) & 0b11111) << std::endl;
    }
    void get_rs2() {
        std::cout << ((this->raw >> 20) & 0b11111) << std::endl;
    }
    void get_rd() {
        std::cout << ((this->raw >> 7) & 0b11111) << std::endl;
    }
    std::string get_op() {
        return "Not implementaed due to unknown instruction type";
    }
};

class AddInst: public RTypeInstruction {
public:
    AddInst(uint32_t raw) : RTypeInstruction(raw) {}
    std::string get_op() {
        return "add op";
    }
};

int main() {
    auto inst = AddInst(0x00728b33);
    inst.get_rs1();
    inst.get_rs2();
    inst.get_rd();
    inst.get_op();
    auto rinst1 = dynamic_cast<RTypeInstruction*>(&inst);
    if (rinst1) {
        // TODO: 填入正确答案
        ASSERT(rinst1->get_op() == "", "get_op() should return " + rinst1->get_op());
    }
    RTypeInstruction rinst2 = inst;
    // TODO: 填入正确答案
    ASSERT(rinst2.get_op() == "", "get_op() should return " + rinst2.get_op());
    return 0;
}