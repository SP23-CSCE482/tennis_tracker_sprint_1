import cv2
import numpy as np
import time
import math

# get mouse click
click_pos = None;
click = False;
def mouseClick(event, x, y, flags, param):
    # hook to globals
    global click_pos;
    global click;

    # check for left mouseclick 
    if event == cv2.EVENT_LBUTTONDOWN:
        click = True;
        click_pos = (x,y);

# return sign of number
def sign(val):
    if val > 0:
        return 1;
    if val < 0:
        return -1;
    return 0;

# create blank image
res = (600,600,3);
bg = np.zeros(res, np.uint8);
display = np.zeros(res, np.uint8);

# set up click callback
cv2.namedWindow("Display");
cv2.setMouseCallback("Display", mouseClick);
click_force = 1000;

# font stuff
font = cv2.FONT_HERSHEY_SIMPLEX;
fontScale = 1;
fontColor = (255, 100, 0);
thickness = 2;

# make a ball
ball_radius = 20;
ball_pos = [300,300];
ball_vel = [0,0];

# set physics
drag = 0.98;
bounce_mult = 0.95;
grav = -9.8; # acceleration in pixels per second
time_scale = 5.0;

# register click animations
click_anims = [];
anim_dur = 0.25; # seconds
anim_radius = 20; # pixels

# track bounces
prev_pos = ball_pos[:];
est_vel = [0,0];
prev_est_vel = [0,0];
bounce_count = 0;
bounce_thresh = 10; # velocity must have a sudden change greater than this magnitude to count
camera_fps = 24; # we'll only take snapshots at this speed
camera_timer = 0; # time since last snapshot
snap = False;
pic_count = 0;

# loop
done = False;
prev_time = time.time();
while not done:
    # refresh display
    display = np.copy(bg);

    # update timestep
    now_time = time.time();
    dt = now_time - prev_time;
    dt *= time_scale;
    prev_time = now_time;

    # update physics
    # position
    ball_pos[0] += ball_vel[0] * dt;
    ball_pos[1] += ball_vel[1] * dt;

    # velocity
    ball_vel[1] -= grav * dt;
    drag_mult = (1 - ((1 - drag) * dt));
    ball_vel[0] *= drag_mult;
    ball_vel[1] *= drag_mult;

    # check for mouse click
    if click:
        # register animation
        click = False;
        click_anims.append([time.time(), click_pos[:]]);

        # get dist
        dx = ball_pos[0] - click_pos[0];
        dy = ball_pos[1] - click_pos[1];
        dist = math.sqrt(dx*dx + dy*dy);

        # clamp dist
        if dist < 1:
            dist = 1;

        # get force attenuation
        # force = click_force / (dist*dist); # too much
        force = click_force / dist; 

        # get angle and get axial force
        angle = math.atan2(dy, dx);
        xforce = math.cos(angle) * force;
        yforce = math.sin(angle) * force;

        # apply force
        ball_vel[0] += xforce;
        ball_vel[1] += yforce;


    # check for bounce
    # left
    if ball_pos[0] - ball_radius < 0:
        ball_pos[0] = 0 + ball_radius;
        ball_vel[0] *= -bounce_mult;

    # right
    if ball_pos[0] + ball_radius > res[0]:
        ball_pos[0] = res[0] - ball_radius;
        ball_vel[0] *= -bounce_mult;

    # up # +y-axis is down in OpenCV
    if ball_pos[1] - ball_radius < 0:
        ball_pos[1] = 0 + ball_radius;
        ball_vel[1] *= -bounce_mult;

    # down
    if ball_pos[1] + ball_radius > res[1]:
        ball_pos[1] = res[1] - ball_radius;
        ball_vel[1] *= -bounce_mult;

    # check if it's time for a snapshot
    camera_timer += dt; # time since last snapshot
    if camera_timer > (1.0 / camera_fps):
        # estimate velocity
        est_vel[0] = (ball_pos[0] - prev_pos[0]) / camera_timer;
        est_vel[1] = (ball_pos[1] - prev_pos[1]) / camera_timer;

        # check if the sign of the velocity has changed
        if sign(est_vel[0]) != sign(prev_est_vel[0]) or sign(est_vel[1]) != sign(prev_est_vel[1]):
            # check for bounces from large change in velocity
            dvx = abs(est_vel[0] - prev_est_vel[0]);
            dvy = abs(est_vel[1] - prev_est_vel[1]);
            change_vel = math.sqrt(dvx*dvx + dvy*dvy);
            if change_vel > bounce_thresh:
                bounce_count += 1;

        # update previous state trackers
        prev_est_vel = est_vel[:];
        prev_pos = ball_pos[:];

        # reset camera timer
        camera_timer = 0;
        snap = True;

    # draw bounce text
    cv2.putText(display, "Bounces: " + str(bounce_count), (15,40), font,
                fontScale, fontColor, thickness, cv2.LINE_AA);

    # draw ball
    x, y = ball_pos;
    cv2.circle(display, (int(x), int(y)), ball_radius, (220,150,0), -1);

    # draw click animations
    for a in range(len(click_anims)-1, -1, -1):
        # get lifetime
        life = now_time - click_anims[a][0];
        if life > anim_dur:
            del click_anims[a];
        else:
            # draw
            mult = life / anim_dur;
            radius = int(anim_radius * mult);
            if radius > 0:
                val = 255 - int(255 * mult);
                color = [val, val, val];
                cv2.circle(display, click_anims[a][1], radius, color, 2);

    # show
    cv2.imshow("Display", display);
    key = cv2.waitKey(1);

    # if snapshot, save a picture
    if snap:
      snap = False;
      cv2.imwrite(str(pic_count).zfill(5) + ".png", display);
      pic_count += 1;

    # check keypresses
    done = key == ord('q');