// tq2_main.cpp — aiesim/x86sim testbench: run the graph once.
#include "tq2_graph.h"

Tq2Graph g;

int main() {
    g.init();
    g.run(1);
    g.end();
    return 0;
}
