CC=nvc

CFLAGS= -g -O0 -mp -acc -acc=noautopar -gpu=cc61 -gpu=cuda12.9 -Minfo=all

# Assume all .c files correspond to a program to build.
TARGETS=$(patsubst %.c,%,$(wildcard *.c))

all: $(TARGETS)

# This is a static pattern rule to build all .c programs (TARGETS).
# Don't define a recipe. Instead, rely on implicit rules to build these. If
# we need to give more specific recipes, we can override the implicit rule
# by writing a recipe for the target.
$(TARGETS): %: %.c
	$(CC) $(CFLAGS) -o $@.x $<

clean:
	rm -f $(addsuffix .x, $(TARGETS))
