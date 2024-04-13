#include <stdio.h>
#include <string.h>

void vulnerableFunction(const char *input) {
    char buffer[10];
    strcpy(buffer, input); 
}

int main() {
    char userInput[20];
    printf("Введите строку: ");
    scanf("%s", userInput);
    vulnerableFunction(userInput);

    return 0;
}