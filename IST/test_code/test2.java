public class Test {

    // For style 21: recursive/iterative
    public static int factorialRecursive(int n) {
        if (n == 0) {
            return 1;
        }
        return n * factorialRecursive(n - 1);
    }

    // For style 17: nested if
    public static void checkNested(int a, int b) {
        if (a > 0 && b > 0) {
            System.out.println("Both positive");
        }
    }

    // For style 16: if/switch
    public static void printDay(int day) {
        switch (day) {
            case 1:
                System.out.println("Monday");
                break;
            case 2:
                System.out.println("Tuesday");
                break;
            default:
                System.out.println("Other day");
                break;
        }
    }

    public static void main(String[] args) {
        // For style 7, 9: declaration split/merge, assign split/merge
        int a;
        a = 5;
        int b = 10;
        int myVar = 20; // For style 0: identifier name

        // For style 2: augmented assignment
        a = a + 1;
        b += 5;

        // For style 1, 11, 12: brackets, for/while, infinite loop
        for (int i = 0; i < 5; i++) {
            a++; // For style 4: for update
        }

        int j = 0;
        while (j < 3) {
            b--;
            j++;
        }

        // For style 19: ternary
        int maxVal = (a > b) ? a : b;

        // For style 14: if exclamation
        if (!(a == b)) {
            System.out.println("Not equal");
        }

        // For style 13: break/goto (Java has no goto)
        for (int k = 0; k < 10; k++) {
            if (k == 3) {
                break;
            }
        }

        // For style 8: declare position
        int tempVar = a + b;
        System.out.println("Result: " + tempVar);
    }
}
