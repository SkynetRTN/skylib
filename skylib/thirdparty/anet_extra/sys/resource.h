
#ifndef _SYS_RESOURCE_H
#define _SYS_RESOURCE_H

#include <stdint.h>
#include <time.h>

typedef uint64_t	rlim_t;

struct rlimit {
	rlim_t	rlim_cur;		/* current (soft) limit */
	rlim_t	rlim_max;		/* maximum value for rlim_cur */
};

struct	rusage {
	struct timeval ru_utime;	/* user time used (PL) */
	struct timeval ru_stime;	/* system time used (PL) */
	long	ru_maxrss;		/* max resident set size (PL) */
	long	ru_ixrss;		/* integral shared memory size (NU) */
	long	ru_idrss;		/* integral unshared data (NU)  */
	long	ru_isrss;		/* integral unshared stack (NU) */
	long	ru_minflt;		/* page reclaims (NU) */
	long	ru_majflt;		/* page faults (NU) */
	long	ru_nswap;		/* swaps (NU) */
	long	ru_inblock;		/* block input operations (atomic) */
	long	ru_oublock;		/* block output operations (atomic) */
	long	ru_msgsnd;		/* messages sent (atomic) */
	long	ru_msgrcv;		/* messages received (atomic) */
	long	ru_nsignals;		/* signals received (atomic) */
	long	ru_nvcsw;		/* voluntary context switches (atomic) */
	long	ru_nivcsw;		/* involuntary " */
};

#define	RLIM_INFINITY	(((uint64_t)1 << 63) - 1)	/* no limit */

#define	RLIMIT_NOFILE	8		/* number of open files */

#define	RUSAGE_SELF	0		/* Current process information */

int	getrlimit(int resource, struct rlimit *rlim);
int	setrlimit(int resource, const struct rlimit *rlim);
int	getrusage(int who, struct rusage *usage);

#endif
