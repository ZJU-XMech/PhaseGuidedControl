#include <main_helper.h>
#include "CPG_Controller.hpp"

int main(int argc, char** argv){
    main_helper(argc, argv, new CPG_Controller());
    return 0;
}