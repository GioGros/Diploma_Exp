#include <stdio.h>
#include <stdlib.h>

int main() {
    // Неправильное выделение памяти
    int *ptr = (int *)malloc(5 * sizeof(int)); 
    ptr[5] = 10; 

    
    free(ptr); 
    free(ptr); 

    
    int *newPtr = (int *)malloc(3 * sizeof(int));
    free(newPtr);
    newPtr[0] = 5; 

    return 0;
}
