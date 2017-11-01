#pragma once


__host__ __device__
inline unsigned int utilhash(unsigned int val) {
	val = (val + 0x7ed55d16) + (val << 12);
	val = (val ^ 0xc761c23c) ^ (val >> 19);
	val = (val + 0x165667b1) + (val << 5);
	val = (val + 0xd3a2646c) ^ (val << 9);
	val = (val + 0xfd7046c5) + (val << 3);
	val = (val ^ 0xb55a4f09) ^ (val >> 16);
	return val;
}