// stq_main.cpp — aiesim/x86sim testbench: run the graph once.
#include "stq_graph.h"

StqGraph g;

int main() {
    g.init();
    g.run(1);
    g.end();
    return 0;
}
