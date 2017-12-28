#include "jsonResolve.h"
#include "cutil_math.h"


float3 resolveFloat3(const Json::Value& jval) {
	if (!jval.isArray())
		return make_float3(0);
	if (jval.size() == 3)
		return make_float3(jval[0].asFloat(), jval[1].asFloat(), jval[2].asFloat());
	if (jval.size() == 1)
		return make_float3(jval[0].asFloat());
	return make_float3(0);
}

glm::vec3 resolveGlmVec3(const Json::Value& jval) {
	if (!jval.isArray())
		return glm::vec3(0);
	if (jval.size() == 3)
		return glm::vec3(jval[0].asFloat(), jval[1].asFloat(), jval[2].asFloat());
	if (jval.size() == 1)
		return glm::vec3(jval[0].asFloat());
	return glm::vec3(0);
}

float resolveFloat(const Json::Value& jval) {
	return jval.asFloat();
}

MaterialType resolveMatType(const Json::Value& jval) {
	std::string t = jval.asString();
	return resolveMatType(t);
}

MaterialType resolveMatType(const std::string& t) {
	if (!t.compare("diff")) {
		return Diffuse;
	}
	else if (!t.compare("spec")) {
		return Specular;
	}
	else if (!t.compare("trans")) {
		return Transparent;
	}
	else if (!t.compare("lumi")) {
		return Luminescent;
	}
	else return Diffuse;
}
