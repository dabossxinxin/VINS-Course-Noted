#pragma once

#include <iostream>

class Context {
public:
	Context();
	Context(const Context&) = delete;
	void operator=(const Context&) = delete;

	virtual ~Context();

	// Creates a context object and the caller takes the ownership 
	static Context* Create();
};