#include <stdio.h>
#include <string.h>

void vulnerableFunction(const char *input) {
    char buffer[10];
    strcpy(buffer, input); 
}

int main() {
    char userInput[20];
    printf("Введите строку: ");
    gets(userInput); 

    vulnerableFunction(userInput);

    printf("Программа завершена\n");
    return 0;
}
