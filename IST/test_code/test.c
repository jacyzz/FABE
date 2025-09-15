#include <stdio.h>
#include <stdlib.h>

// For style 21: recursive/iterative
int factorial_recursive(int n) {
    if (n == 0) {
        return 1;
    }
    return n * factorial_recursive(n - 1);
}

// For style 17: nested if
void check_nested(int a, int b) {
    if (a > 0 && b > 0) {
        printf("Both positive\n");
    }
}

// For style 16: if/switch
void print_day(int day) {
    if (day == 1) {
        printf("Monday\n");
    } else if (day == 2) {
        printf("Tuesday\n");
    } else {
        printf("Other day\n");
    }
}

int main() {
    // For style 7, 9: declaration split/merge, assign split/merge
    int a;
    a = 5;
    int b = 10;
    int my_var = 20; // For style 0: identifier name

    // For style 2: augmented assignment
    a = a + 1;
    b += 5;

    // For style 1, 11, 12: brackets, for/while, infinite loop
    for (int i = 0; i < 5; i++) {
        a++; // For style 4: for update
    }

    int j = 0;
    while(j < 3) {
        b--;
        j++;
    }

    // For style 19: ternary
    int max_val = (a > b) ? a : b;

    // For style 14: if exclamation
    if (!(a == b)) {
        printf("Not equal\n");
    }

    // For style 5, 6: array definition and access
    int static_array[5];
    for(int i=0; i<5; i++) {
        static_array[i] = i;
    }

    int* dynamic_array = (int*)malloc(5 * sizeof(int));
    if (dynamic_array != NULL) {
        *(dynamic_array + 0) = 100;
        free(dynamic_array);
    }

    // For style 13: break/goto
    for(int k=0; k<10; k++) {
        if (k == 3) break;
    }

    // For style 8: declare position
    int temp_var = a + b;
    printf("Result: %d\n", temp_var);

    return 0;
}