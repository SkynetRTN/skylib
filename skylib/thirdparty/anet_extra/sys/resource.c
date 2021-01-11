
#include "resource.h"

__inline int	getrlimit(int resource, struct rlimit *rlim) {
	rlim->rlim_cur = RLIM_INFINITY;
	rlim->rlim_max = RLIM_INFINITY;
	return 0;
}

__inline int	setrlimit(int resource, const struct rlimit *rlim) { return 0; }

__inline int	getrusage(int who, struct rusage *usage) {
	usage->ru_utime.tv_sec = 0;
	usage->ru_utime.tv_usec = 0;
	usage->ru_stime.tv_sec = 0;
	usage->ru_stime.tv_usec = 0;
	return 0;
}
