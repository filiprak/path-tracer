#include "utility.h"


/* based on: http://www.algolist.net/Algorithms/Sorting/Quicksort */
void fquicksort(float arr[], int left, int right) {
	int i = left, j = right;
	float tmp;
	float pivot = arr[(left + right) / 2];

	/* partition */
	while (i <= j) {
		while (arr[i] < pivot)
			i++;
		while (arr[j] > pivot)
			j--;
		if (i <= j) {
			tmp = arr[i];
			arr[i] = arr[j];
			arr[j] = tmp;
			i++;
			j--;
		}
	};

	/* recursion */
	if (left < j)
		fquicksort(arr, left, j);
	if (i < right)
		fquicksort(arr, i, right);
}


float fmedian(float arr[], int size) {
	if (size < 1)
		return 0.0f;
	fquicksort(arr, 0, size - 1);
	int halfs = size / 2;
	if (size % 2 == 0)
		return ((arr[halfs - 1] + arr[halfs]) / 2.0f);
	return arr[halfs];
}