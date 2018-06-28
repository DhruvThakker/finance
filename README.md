# Computational Finance
Repository for ReST APIs
postgres://USER:PASSWORD@HOST:PORT/NAME
https://simpleisbetterthancomplex.com/tutorial/2016/08/09/how-to-deploy-django-applications-on-heroku.html
https://devcenter.heroku.com/articles/heroku-postgresql#local-setup

heroku config:set DISABLE_COLLECTSTATIC=1 -a computational-finance-api
pip freeze > requirements.txt
heroku login
heroku apps
heroku addons:create heroku-postgresql:hobby-dev
heroku create computational-finance-api

sudo -u postgres createuser remedy441 -d -P
su remedy441
createdb remedy441

local python-decouple, dj-database-url,.env.example
deply gunicorn,runtime,Procfile,wsgi.py