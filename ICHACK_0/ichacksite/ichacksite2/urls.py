from django.urls import path

from . import views
from .access import authenticated_view, teacher_required

urlpatterns = [
    path("", views.index, name="index"),
    path("login", views.login_route, name="login_route"),
    path("about", views.about, name="about"),
    path("team", views.team, name="team"),
    path("loginUser", views.loginUser, name="loginUser"),
    path(
        "getDetailsForCourse",
        authenticated_view(views.getDetailsForCourse),
        name="getDetailsForCourse",
    ),
    path(
        "getEssayQuestions",
        authenticated_view(views.getEssayQuestions),
        name="getEssayQuestions",
    ),
    path(
        "getQuestionsForCourse_teacher",
        teacher_required(views.getQuestionsForCourse_teacher),
        name="getQuestionsForCourse_teacher",
    ),
    path(
        "getSubmissionsForQuestion_teacher",
        teacher_required(views.getSubmissionsForQuestion_teacher),
        name="getSubmissionsForQuestion_teacher",
    ),
    path(
        "add_submission",
        authenticated_view(views.add_submission),
        name="add_submission",
    ),
]
