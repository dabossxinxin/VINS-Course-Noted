#include "utility/context.h"
#include "utility/context_impl.h"

Context::Context() = default;

Context* Context::Create() {
	return new ContextImpl();
}

Context::~Context() = default;