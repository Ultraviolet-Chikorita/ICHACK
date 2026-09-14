from functools import wraps

from django.contrib.auth.decorators import login_required
from django.http import HttpResponseForbidden
from django.views.decorators.csrf import csrf_protect


def authenticated_view(view_func):
    """Require an authenticated session and normal Django CSRF validation."""
    return csrf_protect(login_required(view_func))


def teacher_required(view_func):
    """Restrict educator-only data endpoints to teachers or superusers."""

    @wraps(view_func)
    def guarded(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return login_required(view_func)(request, *args, **kwargs)
        if not (request.user.is_teacher or request.user.is_superuser):
            return HttpResponseForbidden("Teacher access required")
        return view_func(request, *args, **kwargs)

    return csrf_protect(guarded)
