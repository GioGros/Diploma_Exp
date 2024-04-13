#include <stdio.h>

int main() {
    char userInput[50];
    printf("Введите строку: ");
    scanf("%s", userInput); 

    printf("Вы ввели: ");
    printf(userInput); 

    printf("\nПрограмма завершена\n");
    return 0;
}
