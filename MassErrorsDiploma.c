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
    if (NULL) { 
        printf("Эта строка не должна выводиться\n");
    }

    int x = 10;
    if (x = 20) { 
        printf("Эта строка не должна выводиться\n");
    }

    int arr[5];
    for (int i = 0; i <= 5; i++) { 
        arr[i] = i;
    }

    int y = 0;
    if (y = 1) { 
        printf("Эта строка не должна выводиться\n");
    }

    sizeof(int); 

    if (x > 5 && x < 15) { 
        printf("Эта строка не должна выводиться\n");
    }

    vulnerableFunction(userInput);

    return 0;
}