#include <stdio.h>

int main() {
    int x = 10;
    int y = 0;
    int z;

    // Деление на ноль
    z = x / y;

    // Неиспользуемая переменная
    int unused_variable;

    // Операция сравнения вместо присваивания
    if (x == 5) {
        printf("x is 5\n");
    }

    return 0;
}