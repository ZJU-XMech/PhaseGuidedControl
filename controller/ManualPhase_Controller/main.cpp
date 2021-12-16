#include <main_helper.h>
#include "ManualPhase_Controller.hpp"

int main(int argc, char** argv){
    main_helper(argc, argv, new MP_Controller());
    return 0;
}