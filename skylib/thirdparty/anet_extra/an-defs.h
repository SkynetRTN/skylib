
#ifndef _AN_DEFS_H
#define _AN_DEFS_H

#include <sys/types.h>
#include <sys/stat.h>
#include <stdint.h>
#include <windows.h>
#include <pthread.h>
#include <io.h>

typedef unsigned int uint;

typedef uint32_t uid_t;

union sigval {
	int	sival_int;
	void	*sival_ptr;
};

struct sigevent {
	int				sigev_notify;				/* Notification type */
	int				sigev_signo;				/* Signal number */
	union sigval	sigev_value;				/* Signal value */
	void			(*sigev_notify_function)(union sigval);	  /* Notification function */
	pthread_attr_t	*sigev_notify_attributes;	/* Notification attributes */
};

typedef struct __siginfo {
	int	si_signo;		/* signal number */
	int	si_errno;		/* errno association */
	int	si_code;		/* signal code */
	pid_t	si_pid;			/* sending process */
	uid_t	si_uid;			/* sender's ruid */
	int	si_status;		/* exit value */
	void	*si_addr;		/* faulting instruction */
	union sigval si_value;		/* signal value */
	long	si_band;		/* band event for SIGPOLL */
	unsigned long	__pad[7];	/* Reserved for Future Use */
} siginfo_t;

struct	sigaction {
	void    (*sa_handler)(int);
	sigset_t sa_mask;
	int	sa_flags;
};

#undef mkdir
#define mkdir(A, B) _mkdir(A)

#define	S_IFLNK		0120000		/* [XSI] symbolic link */
#define	S_ISLNK(m)	(((m) & S_IFMT) == S_IFLNK)	/* symbolic link */

int sigaction(int, const struct sigaction *, struct sigaction *);
char *mkdtemp(char *);
int fsync(int);
int pipe(int [2]);
pid_t fork(void);
char *realpath(const char *, char *);

#endif
