from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('login', views.login_route, name='login_route'),
    path('about', views.about, name='about'),
    path('team', views.team, name='team'),
    path('loginUser', views.loginUser, name='loginUser'),
    path('getDetailsForCourse', views.getDetailsForCourse, name='getDetailsForCourse'),
    path('getEssayQuestions', views.getEssayQuestions, name='getEssayQuestions'),
    path('getQuestionsForCourse_teacher', views.getQuestionsForCourse_teacher, name='getQuestionsForCourse_teacher'),
    path('getSubmissionsForQuestion_teacher', views.getSubmissionsForQuestion_teacher, name='getSubmissionsForQuestion_teacher'),
    path('add_submission', views.add_submission, name="add_submission")
]