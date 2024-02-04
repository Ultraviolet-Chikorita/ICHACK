from django.db import models
from django.urls import reverse
from django.contrib.auth.models import AbstractUser

# Create your models here.

class CourseName(models.Model):
    name = models.CharField(
        max_length = 200,
        help_text = "Enter course name",
        unique = True
    )

    def __str__(self):
        return self.name

class User(AbstractUser):
    is_teacher = models.BooleanField('teacher status', default=False)
    courses = models.ManyToManyField(CourseName, related_name='user_groups', blank=True)

    def __str__(self):
        return self.email

class TaskBank(models.Model):
    question = models.CharField(
        max_length = 2000
    )
    course = models.ForeignKey(CourseName, on_delete=models.CASCADE)
    date_set = models.DateField()
    date_due = models.DateField()

class Submission(models.Model):
    essay = models.CharField(
        max_length = 40000,
        help_text = "Essay submitted as homework"
    )
    user = models.ForeignKey(
        User, 
        on_delete=models.CASCADE
    )
    prompt = models.ForeignKey(
        TaskBank, 
        on_delete=models.CASCADE,
        related_name="submissions"
    )
    question1 = models.CharField(
        max_length = 2000,
        help_text = "Enter question 1"
    )
    question2 = models.CharField(
        max_length = 2000,
        help_text = "Enter question 2"
    )
    question3 = models.CharField(
        max_length = 2000,
        help_text = "Enter question 3"
    )
    question4 = models.CharField(
        max_length = 2000,
        help_text = "Enter question 4"
    )
    question5 = models.CharField(
        max_length = 2000,
        help_text = "Enter question 5"
    )
    url1 = models.CharField(
        max_length=300
    )
    url2 = models.CharField(
        max_length=300
    )
    url3 = models.CharField(
        max_length=300
    )
    url4 = models.CharField(
        max_length=300
    )
    url5 = models.CharField(
        max_length=300
    )
    processedurl1 = models.CharField(
        max_length=300
    )
    processedurl2 = models.CharField(
        max_length=300
    )
    processedurl3 = models.CharField(
        max_length=300
    )
    processedurl4 = models.CharField(
        max_length=300
    )
    processedurl5 = models.CharField(
        max_length=300
    )
    gazeSuspicion = models.FloatField() # percentage of time outside ellipse
    polarity = models.FloatField()

    def __str__(self):
        return "Submission " + str(self.id)