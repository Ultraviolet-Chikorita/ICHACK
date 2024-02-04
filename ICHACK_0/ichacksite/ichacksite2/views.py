from datetime import datetime
from django.shortcuts import render
from .models import Submission, CourseName, User, TaskBank 
from django.contrib.auth import authenticate, login, logout
from django.shortcuts import redirect
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
from openai import OpenAI
import mediapipe as mp
import cv2
import numpy as np
from openai import OpenAI
# from os import subprocess

# Create your views here.

relative = lambda landmark, shape: (int(landmark.x * shape[1]), int(landmark.y * shape[0]))
relativeT = lambda landmark, shape: (int(landmark.x * shape[1]), int(landmark.y * shape[0]), 0)

def speech_sentiment(video):
    client = OpenAI()
    transcript_list = []
    for i in range(3):
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=video,
            temperature=0.8,
            response_format="text",
            prompt="Try to capture every thing, including 'uh', 'ummm', or 'ah'. Any audible pauses in the middle of speaking you should replace with '...'"
        )
        transcript_list.append(transcript)
    from transformers import pipeline
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k = None)
    prediction_list = []
    for i in range(3):
        prediction = classifier(transcript_list[i], )
        prediction_list.append(prediction)
    emotion_scores = {'anger': 0, 'neutral': 0, 'fear': 0, 'surprise': 0, 'sadness': 0, 'disgust': 0, 'joy': 0}
    print(prediction_list)
    for i in range(3):
        for j in range(len(prediction_list)):
            emotion = prediction_list[i][0][j]['label']
            score = prediction_list[i][0][j]['score']
            emotion_scores[emotion] += score
    for emotion in emotion_scores:
        emotion_scores[emotion] /= 3
    classifier2 = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=True)
    prediction_list_2 = []
    for i in range(3):
        prediction = classifier2(transcript_list[i], )
        prediction_list_2.append(prediction)
    emotion_scores_2 = {'anger': 0, 'fear': 0, 'surprise': 0, 'sadness': 0, 'love': 0, 'joy': 0}
    for i in range(3):
        for j in range(len(prediction_list_2[i])):
            emotion = prediction_list_2[i][0][j]['label']
            score = prediction_list_2[i][0][j]['score']
            emotion_scores_2[emotion] += score
    for emotion in emotion_scores_2:
        emotion_scores_2[emotion] /= 3
    neutral_score = emotion_scores['neutral']
    negative_score = (emotion_scores['disgust'] * 1.2 + emotion_scores['fear'] * 1.3 + emotion_scores_2['fear'] * 1.3 + emotion_scores['sadness'] * 0.6 + emotion_scores['sadness'] * 0.6) / 4.0
    positive_score = (emotion_scores['joy'] * 0.8 + emotion_scores_2['love'] * 1.4 + emotion_scores_2['joy'] * 0.8) / 3.0
    polarity = neutral_score + 0.5 * (abs(positive_score - negative_score))
    return polarity

def gaze(frame, points):

    """

    The gaze function gets an image and face landmarks from mediapipe framework.

    The function draws the gaze direction into the frame.

    """

 

    '''

    2D image points.

    relative takes mediapipe points that is normalized to [-1, 1] and returns image points

    at (x,y) format

    '''

    image_points = np.array([

        relative(points.landmark[4], frame.shape),  # Nose tip

        relative(points.landmark[152], frame.shape),  # Chin

        relative(points.landmark[263], frame.shape),  # Left eye left corner

        relative(points.landmark[33], frame.shape),  # Right eye right corner

        relative(points.landmark[287], frame.shape),  # Left Mouth corner

        relative(points.landmark[57], frame.shape)  # Right mouth corner

    ], dtype="double")

 

    '''

    2D image points.

    relativeT takes mediapipe points that is normalized to [-1, 1] and returns image points

    at (x,y,0) format

    '''

    image_points1 = np.array([

        relativeT(points.landmark[4], frame.shape),  # Nose tip

        relativeT(points.landmark[152], frame.shape),  # Chin

        relativeT(points.landmark[263], frame.shape),  # Left eye, left corner

        relativeT(points.landmark[33], frame.shape),  # Right eye, right corner

        relativeT(points.landmark[287], frame.shape),  # Left Mouth corner

        relativeT(points.landmark[57], frame.shape)  # Right mouth corner

    ], dtype="double")

 

    # 3D model points.

    model_points = np.array([

        (0.0, 0.0, 0.0),  # Nose tip

        (0, -63.6, -12.5),  # Chin

        (-43.3, 32.7, -26),  # Left eye, left corner

        (43.3, 32.7, -26),  # Right eye, right corner

        (-28.9, -28.9, -24.1),  # Left Mouth corner

        (28.9, -28.9, -24.1)  # Right mouth corner

    ])

 

    '''

    3D model eye points

    The center of the eye ball

    '''

    Eye_ball_center_right = np.array([[-29.05], [32.7], [-39.5]])

    Eye_ball_center_left = np.array([[29.05], [32.7], [-39.5]])  # the center of the left eyeball as a vector.

 

    '''

    camera matrix estimation

    '''

    focal_length = frame.shape[1]

    center = (frame.shape[1] / 2, frame.shape[0] / 2)

    camera_matrix = np.array(

        [[focal_length, 0, center[0]],

         [0, focal_length, center[1]],

         [0, 0, 1]], dtype="double"

    )

 

    #dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    dist_coeffs = np.array([[-1.76826357e-01],  [1.65233067e+01], [-1.19047008e-03],[3.18442756e-03]])

    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,

                                                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

 

    # 2d pupil location

    left_pupil = relative(points.landmark[468], frame.shape)

    right_pupil = relative(points.landmark[473], frame.shape)

 

    # Transformation between image point to world point

    _, transformation, _ = cv2.estimateAffine3D(image_points1, model_points)  # image to world transformation

 

    if transformation is not None:  # if estimateAffine3D secsseded

        # project pupil image point into 3d world point

        pupil_world_cord = transformation @ np.array([[left_pupil[0], left_pupil[1], 0, 1]]).T

 

        pupil_world_cordr = transformation @ np.array([[right_pupil[0], right_pupil[1], 0, 1]]).T

 

        # 3D gaze point (10 is arbitrary value denoting gaze distance)

        S = Eye_ball_center_left + (pupil_world_cord - Eye_ball_center_left) * 10

 

        Sr = Eye_ball_center_right + (pupil_world_cordr - Eye_ball_center_right) * 10

 

        # Project a 3D gaze direction onto the image plane.

        (eye_pupil2D, _) = cv2.projectPoints((int(S[0]), int(S[1]), int(S[2])), rotation_vector,

                                             translation_vector, camera_matrix, dist_coeffs)

       

        (eye_pupil2Dr, _) = cv2.projectPoints((int(Sr[0]), int(Sr[1]), int(Sr[2])), rotation_vector,

                                             translation_vector, camera_matrix, dist_coeffs)

       

        

        # project 3D head pose into the image plane

        (head_pose, _) = cv2.projectPoints((int(pupil_world_cord[0]), int(pupil_world_cord[1]), int(40)),

                                           rotation_vector,

                                           translation_vector, camera_matrix, dist_coeffs)

       

        (head_poser, _) = cv2.projectPoints((int(pupil_world_cordr[0]), int(pupil_world_cordr[1]), int(40)),

                                           rotation_vector,

                                           translation_vector, camera_matrix, dist_coeffs)

        # correct gaze for head rotation

        gaze = left_pupil + (eye_pupil2D[0][0] - left_pupil) - (head_pose[0][0] - left_pupil)

        gazer = right_pupil + (eye_pupil2Dr[0][0] - right_pupil) - (head_poser[0][0] - right_pupil)

 

        # draw point you are looking at

        #cv2.circle(frame,(int(eye_pupil2D[0][0][0]), int(eye_pupil2D[0][0][1])), 10, (0,0,255), 2)

        #cv2.circle(frame,(int(eye_pupil2Dr[0][0][0]), int(eye_pupil2Dr[0][0][1])), 10, (0,0,255), 2)

 

        # Draw gaze line into screen

        pl1 = (int(left_pupil[0]), int(left_pupil[1]))

        p2 = (int((gaze[0] + gazer[0])/ 2) , int((gaze[1] + gazer[1])/ 2))

        cv2.line(frame, pl1, p2, (0, 0, 255), 2)

 

        #Draw another gaze

        pr1 = (int(right_pupil[0]), int(right_pupil[1]))

        cv2.line(frame, pr1, p2, (0, 0, 255), 2)

 

         # draw point you are looking at

        cv2.circle(frame,(int(gazer[0]), int(gazer[1])), 10, (0,0,255), 2)

        cv2.circle(frame,(int(gaze[0]), int(gaze[1])), 10, (0,0,255), 2)

 

        center = ((left_pupil[0]+ right_pupil[0])//2, (left_pupil[1]+ right_pupil[1])//2)

        axes = (int(frame.shape[1] / 6), int(frame.shape[0] / 8) ) # Width and height of the ellipse

        angle = 0  # Angle of rotation of the ellipse

        startAngle = 0  # Starting angle of the elliptical arc

        endAngle = 360  # Ending angle of the elliptical arc

        color = (255, 0, 0)  # Blue color in BGR

        thickness = 2  # Thickness of the ellipse border

 

        cv2.ellipse(frame, center, axes, angle, startAngle, endAngle, color, thickness)

        #cv2.circle(frame, eye_pupil2D[0][0], 10, (255,0,0), 10)

 

        cv2.imshow('Frame with Ellipse', frame)  # Display the frame

 

        a, b = axes

        x, y = p2

        c1, c2 = center

 

        return 1 if ((((x-c1)**2)/(a**2) + ((y-c2)**2)/(b**2)) > 1) else 0

    return 1


def index(request):
    if request.user.is_authenticated:
        courses = []
        for course in list(request.user.courses.all()):
            courses.append({"id": course.id, "name": course.name})
        if request.user.is_teacher:
            return render(request, 'teacher_index.html', {"courses": courses})
        min_number = 0
        for _ in range(Submission.objects.count()):
            min_number += 5
        return render(request, 'student_index.html', {"courses": courses, "min_number": min_number})
    return redirect('login_route')

def login_route(request):
    if request.user.is_authenticated:
        logout(request)
    return render(request, 'login.html')

def team(request):
    return render(request, 'team.html')

def about(request):
    return render(request, 'about.html')

def loginUser(request):
    if request.method == "POST":
        data = request.POST
        username = data.get("username")
        password = data.get("password")
        user = authenticate(username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('index')
        else:
            return redirect('login_route')

@csrf_exempt
def getDetailsForCourse(request):
    if request.method == "POST":
        data = json.load(request)
        course_id = int(data.get('course'))
        teachers = list(User.objects.filter(courses__id=course_id, is_teacher=True))
        current_date = datetime.today().strftime('%Y-%m-%d')
        teacher_list = []
        for teacher in teachers:
            teacher_list.append({"id": teacher.id, "name": teacher.username})
        questions = list(TaskBank.objects.filter(course__id=course_id, date_set__lte=current_date, date_due__gte=current_date))
        question_list = []
        for question in questions:
            question_list.append({"id":question.id, "q":question.question})
        return JsonResponse({"teachers": teacher_list, "questions": question_list})

@csrf_exempt
def getQuestionsForCourse_teacher(request):
    if request.method == "POST":
        data = json.load(request)
        course_id = int(data.get('course'))
        questions = list(TaskBank.objects.filter(course__id=course_id))
        question_list = []
        for question in questions:
            question_list.append({"id":question.id, "q":question.question})
        return JsonResponse({"questions": question_list})

@csrf_exempt
def getSubmissionsForQuestion_teacher(request):
    if request.method == "POST":
        data = json.load(request)
        question_id = int(data.get('question'))
        submissions = list(TaskBank.objects.filter(id=question_id).first().submissions.all())
        submission_list = []
        for submission in submissions:
            submission_list.append({"id":submission.id, "student": submission.user.username, "prompt":submission.prompt.question, "essay": submission.essay, "questions": [submission.question1, submission.question2, submission.question3, submission.question4, submission.question5], "answers": [submission.url1[18:], submission.url2[18:], submission.url3[18:], submission.url4[18:], submission.url5[18:]], "processed": [submission.processedurl1[18:], submission.processedurl2[18:], submission.processedurl3[18:], submission.processedurl4[18:], submission.processedurl5[18:]], "gazeSuspicion": round(submission.gazeSuspicion, 2), "polarity": 1 - round(submission.polarity, 2)})
        return JsonResponse({"submissions": submission_list})

@csrf_exempt
def getEssayQuestions(request):
    if request.method == "POST":
        data = json.load(request)
        essay = data.get('essay')
        question = data.get('question')
        client = OpenAI()
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages= [
                {"role": "system", "content": f'For this prompt:\n {question}\nThe following essay was written:\n{essay}'},
                {"role": "user", "content": "Generate 5 questions about the essay based on the essay in the context of the prompt and separate each question with a new line."}
            ]
        )
        return JsonResponse({"questions": completion.choices[0].message.content.split('\n')})
        
@csrf_exempt
def add_submission(request):
    if request.method == "POST":
        current_course = request.POST.get('currentCourse')
        current_teacher = request.POST.get('currentTeacher')
        questions_json = request.POST.get('questions')
        essay = request.POST.get('essay')
        prompt = request.POST.get('prompt')
        # Convert JSON string to Python list
        questions = json.loads(questions_json)
        import os
        media_directory = 'ichacksite2/media/'
        if not os.path.exists(media_directory):
            os.makedirs(media_directory)

        # Process and save files
        urls = []
        processedurls = []

        suspicions = []
        polarities = []
        for i, file in enumerate(request.FILES.getlist('videos')):
            file_path = f'ichacksite2/media/{file.name}'
            
            with open(file_path, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)
            
            # CALL HERE
            os.system(f"ffmpeg -fflags +genpts -i {file_path} -r 24 {file_path[:-5]}.mp4")
            mp_face_mesh = mp.solutions.face_mesh
            processedfilepath = f'{file_path[:-5]}.mp4'
            polarities.append(speech_sentiment(open(processedfilepath, "rb")))
            urls.append(processedfilepath)
            extraprocessurl = f'ichacksite2/media/processed_{file.name[:-5]}.mp4'
            processedurls.append(extraprocessurl)
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            out = cv2.VideoWriter(extraprocessurl,fourcc, 20.0, (640,480))
            cap = cv2.VideoCapture(file_path)  # chose camera index (try 1, 2, 3)
            curr_sum = 0
            total_frames=0
            with mp_face_mesh.FaceMesh(
                    max_num_faces=1,  # number of faces to track in each frame
                    refine_landmarks=True,  # includes iris landmarks in the face mesh model
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as face_mesh:
                
                while cap.isOpened():
                    success, image = cap.read()
                    if not success:  # no frame input
                        cv2.destroyAllWindows()
                        print("Ignoring empty camera frame.")
                        break
                    # To improve performance, optionally mark the image as not writeable to
                    # pass by reference.
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # frame to RGB for the face-mesh model
                    results = face_mesh.process(image)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # frame back to BGR for OpenCV
                    if results.multi_face_landmarks:
                        curr_sum += gaze(image, results.multi_face_landmarks[0])  # gaze estimation
                    total_frames += 1
                    image = cv2.flip(image,1)
                    cv2.imshow('output window', image)
                    out.write(image)
                    if cv2.waitKey(2) & 0xFF == 27:
                        break
             
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            gaze_suspicion = float(curr_sum) / float(total_frames)
            suspicions.append(gaze_suspicion)
        overall_sus = float(sum(suspicions)) / 5.0
        submission = Submission(essay=essay, user=User.objects.filter(id=request.user.id).first(), prompt=TaskBank.objects.filter(id=prompt).first(), question1=questions[0], question2=questions[1], question3=questions[2], question4=questions[3], question5=questions[4], url1=urls[0], url2=urls[1], url3=urls[2], url4=urls[3], url5=urls[4], processedurl1=processedurls[0], processedurl2=processedurls[1], processedurl3=processedurls[2], processedurl4=processedurls[3], processedurl5=processedurls[4], gazeSuspicion=overall_sus, polarity=(float(sum(polarities))/5.0))
        submission.save()
        return JsonResponse({"success": "true"})