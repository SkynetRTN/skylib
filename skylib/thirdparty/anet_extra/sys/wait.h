
#ifndef _SYS_WAIT_H
#define _SYS_WAIT_H

#include <sys/types.h>

#define WNOHANG		0x00000001  /* [XSI] no hang in wait/no child to reap */
#define WUNTRACED	0x00000002  /* [XSI] notify on stop, untraced child */

#define	_W_INT(i)	(i)

#define	_WSTATUS(x)	(_W_INT(x) & 0177)
#define	_WSTOPPED	0177		/* _WSTATUS if process is stopped */

/*
 * [XSI] The <sys/wait.h> header shall define the following macros for
 * analysis of process status values
 */
#define WEXITSTATUS(x)	(_W_INT(x) >> 8)
#define WSTOPSIG(x)	(_W_INT(x) >> 8)
#define WIFCONTINUED(x) (_WSTATUS(x) == _WSTOPPED && WSTOPSIG(x) == 0x13)
#define WIFSTOPPED(x)	(_WSTATUS(x) == _WSTOPPED && WSTOPSIG(x) != 0x13)
#define WIFEXITED(x)	(_WSTATUS(x) == 0)
#define WIFSIGNALED(x)	(_WSTATUS(x) != _WSTOPPED && _WSTATUS(x) != 0)
#define WTERMSIG(x)	(_WSTATUS(x))

/*
 * [XSI] The following symbolic constants shall be defined as possible
 * values for the fourth argument to waitid().
 */
/* WNOHANG already defined for wait4() */
/* WUNTRACED defined for wait4() but not for waitid() */
#define	WEXITED		0x00000004  /* [XSI] Processes which have exitted */
#define	WCONTINUED	0x00000010  /* [XSI] Any child stopped then continued */
#define	WNOWAIT		0x00000020  /* [XSI] Leave process returned waitable */

pid_t waitpid(pid_t pid, int *status, int options) { return 0; }

#endif
