#include "main.h"
#include "view.h"

int main() {
	printf("Starting path-tracer application.\n");

	viewInit(300, 300);

	viewLoop();

	exit(0);
}