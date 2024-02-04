from django.contrib import admin
from .models import Submission, CourseName, User, TaskBank

# Register your models here.

@admin.register(Submission)
class submissionAdmin(admin.ModelAdmin):
    pass

@admin.register(CourseName)
class courseNameAdmin(admin.ModelAdmin):
    pass

@admin.register(User)
class userAdmin(admin.ModelAdmin):
    pass

@admin.register(TaskBank)
class taskBankAdmin(admin.ModelAdmin):
    pass
