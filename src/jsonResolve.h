#pragma once

#include "world.h"


float3 resolveFloat3(const Json::Value&);
glm::vec3 resolveGlmVec3(const Json::Value&);
float resolveFloat(const Json::Value&);
MaterialType resolveMatType(const Json::Value&);
