
#include "an-defs.h"

int sigaction(int signum, const struct sigaction *act, struct sigaction *oldact)
{
	return 0;
}

char *mkdtemp(char *template)
{
	return template;
}

int fsync(int fd)
{
	return 0;
}

int pipe(int pipefd[2])
{
	return 0;
}

pid_t fork(void)
{
	return 0;
}

char *realpath(const char *p1, char *p2)
{
	return p1;
}
	