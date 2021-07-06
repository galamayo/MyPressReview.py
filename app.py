# -- coding: utf-8

from flask import Flask, request, Response, jsonify, flash, get_flashed_messages
from flask_compress import Compress
from flask_restless import APIManager
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy import Column, Sequence, ForeignKey, Integer, Text, DateTime, Float, Boolean
from sqlalchemy.sql import insert, select, update, and_, desc
from sqlalchemy.sql.expression import func

# import requests
import sys
import os
import socket
import http.cookiejar
import urllib.request
import urllib.error
import urllib.parse
from ssl import SSLError
from urllib.parse import urljoin
import html.parser
import regex as re
import time
import datetime
import json
from collections import defaultdict, Counter
from enum import Enum
import threading
import gzip

# import unicodedata
import unidecode
import string
import Stemmer
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from marisa_vectorizers import MarisaCountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import joblib


# from proxy_views import proxy


# #########################################################
#
#   Enum types
#


class Learn(Enum):
    LOAD = 0
    NEED = 1
    LEARNING = 2
    OK = 3
    NOK = 4


# #########################################################
#
#   Flask initialisation
#

app = Flask(__name__, static_url_path='')
# app.register_blueprint(proxy)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgres://py:pypass@localhost:5432/journal' #journal'
# app.config['SQLALCHEMY_ECHO'] = True
# app.config['SQLALCHEMY_POOLCLASS'] = QueuePool
app.config['COMPRESS_DEBUG'] = True
app.config['COMPRESS_LEVEL'] = 9
app.config['SECRET_KEY'] = 'MyDGNews-secret-key'
db = SQLAlchemy(app)
Compress(app)


# #########################################################
#
#   ORM mapping
#

class Categories(db.Model):
    id = Column(Integer, Sequence('category_id_seq'), primary_key=True)
    sequence = Column(Integer, unique=False)
    title = Column(Text, nullable=False, unique=False)


class Sources(db.Model):
    id = Column(Integer, Sequence('source_id_seq'), primary_key=True)
    title = Column(Text, nullable=False, unique=False)
    url = Column(Text, nullable=True, unique=False)
    re_list = Column(Text, unique=False)
    re_article = Column(Text, unique=False)
    date_format = Column(Text, unique=False)
    locale = Column(Text, unique=False, default='C')
    login_script = Column(Text, unique=False, default='')
    post_process = Column(Text, unique=False, default='')
    active = Column(Text, unique=False, default='0')


class News(db.Model):
    id = Column(Integer, Sequence('news_id_seq'), primary_key=True)
    date = Column(DateTime, nullable=False, unique=False, index=True)
    title = Column(Text, nullable=False, unique=False)
    abstract = Column(Text, unique=False, default='')
    text = Column(Text, unique=False, default='')
    url = Column(Text, nullable=False, unique=True, index=True)
    source_title = Column(Text, nullable=False, unique=False)  # Duplicate source title for archiving
    category_title = Column(Text, unique=False, index=True)  # Duplicate category title for archiving
    tokens = Column(Text, unique=False, index=False)
    auto_cat = Column(Text, unique=False)
    cat_quote = Column(Float, unique=False, default=0.0)
    publication_id = Column(None, ForeignKey('publications.id'), unique=False, index=True)

    def __init__(self, date, title, abstract, text, url, source_title):
        self.date = date
        self.title = title
        self.abstract = abstract
        self.text = text
        self.url = url
        self.source_title = source_title

    def __repr__(self):
        return '<News %d: %r>' % (self.id, self.title)


class Publications(db.Model):
    id = Column(Integer, Sequence('publications_id_seq'), primary_key=True)
    date = Column(DateTime, unique=True)


# #########################################################
#
#   DB creation & initialisation
#

db.create_all()
try:
    db.engine.execute("INSERT INTO categories VALUES (-2, 0, 'OFF TOPIC')")
    db.engine.execute("INSERT INTO categories VALUES (-1, 1, 'DELETED')")
except IntegrityError:
    pass

# #########################################################
#
#   Global constants
#


PORT = 5000

STEMMERS = defaultdict(lambda: 'french')
STEMMERS.update({'en': 'english', 'fr': 'french', 'de': 'german', 'nl': 'dutch', 'es': 'spanish', 'pt': 'portuguese'})

TRANS_TABLE = {key: None for key in string.punctuation}
TRANS_TABLE.update({key: ' ' for key in "-'’""\xa0\n"})
TRANS_TABLE = str.maketrans(TRANS_TABLE)

# #########################################################
#
#   Global variables
#


dbLock = threading.Lock()

app.streamData = {'data': '', 'close': False}  # Used by get_news to stream progress information
learning = Learn.LOAD
target_names = []  # List of current targets know by the classifier

# #########################################################
#
#   Utility functions
#

tags_re = re.compile(r'<[^>]+>')
num_re = re.compile(r'[0-9]')
word_re = re.compile(r'[\w$\-\x80-\xff]+')


def strip_accents(content):
    return unidecode.unidecode(content)


def remove_punctuation(content):
    # # code inspired form
    # # http://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python
    stripped_input = content.translate(TRANS_TABLE)
    return stripped_input


def generate_tokens(news):
    tokens = ''
    if news.text != '' or news.abstract != '':
        text = ' '.join([news.source_title, news.title, news.abstract, news.text])
        text = tags_re.sub(' ', text)  # remove html tags
        text = num_re.sub('', text)  # remove numbers
        text = text.lower()
        text = strip_accents(text)
        text = remove_punctuation(text)

        stemmer = Stemmer.Stemmer(STEMMERS['fr'])

        tokens = text.split(' ')
        tokens = stemmer.stemWords(tokens)
        tokens = [t for t in tokens if len(t) > 1]
        tokens = ' '.join(tokens)
    return tokens


def get_count(q):
    count_q = q.statement.with_only_columns([func.count()]).order_by(None)
    count = q.session.execute(count_q).scalar()
    return count


# #########################################################
#
#   Classifier
#


class_weight = {}
count_vector = MarisaCountVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.90, stop_words='english')
tfidf_transformer = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
clf_onTopic = LinearSVC(C=100, max_iter=20000, class_weight=class_weight)
clf_categories = MultinomialNB(alpha=0.01, fit_prior=True)


def classifier_load():
    global learning
    global target_names
    global count_vector
    global tfidf_transformer
    global clf_onTopic
    global clf_categories

    # Try to load old classifiers
    try:
        target_names = joblib.load('classifiers/target_names.pkl')
        count_vector = joblib.load('classifiers/count_vector.pkl')
        tfidf_transformer = joblib.load('classifiers/tfidf_transformer.pkl')
        clf_onTopic = joblib.load('classifiers/clf_onTopic.pkl')
        clf_categories = joblib.load('classifiers/clf_categories.pkl')
        learning = Learn.OK
    except:
        learning = Learn.NEED


def classifier_save():
    joblib.dump(target_names, 'classifiers/target_names.pkl')
    joblib.dump(count_vector, 'classifiers/count_vector.pkl')
    joblib.dump(tfidf_transformer, 'classifiers/tfidf_transformer.pkl')
    joblib.dump(clf_onTopic, 'classifiers/clf_onTopic.pkl')
    joblib.dump(clf_categories, 'classifiers/clf_categories.pkl')


def classifier_learn():
    global learning
    global target_names
    global class_weight
    global count_vector
    global tfidf_transformer
    global clf_onTopic
    global clf_categories

    if learning != Learn.LEARNING:
        learning = Learn.LEARNING

        print('Learning', flush=True)

        # Get all categories titles
        stmt = select([Categories.title]).where(Categories.title != 'DELETED')
        result = db.engine.execute(stmt)

        target_names = [r[0] for r in result]
        target_names.sort()

        # Get all news tokens and categories
        stmt = db.session.query(News).filter(
            and_(News.publication_id.isnot(None), News.category_title.in_(target_names))).with_entities(News.tokens,
                                                                                                        News.category_title)
        nb_row = get_count(stmt)
        print(nb_row)

        if nb_row < 10:
            print('Not enough examples to learn from.', flush=True)
            learning = Learn.NOK
            return

        result = db.session.query(News).filter(
            and_(News.publication_id.isnot(None), News.category_title.in_(target_names))).with_entities(News.tokens,
                                                                                                        News.category_title)

        data, y_train_2 = zip(*result)
        y_train_1 = [bool(topic != 'OFF TOPIC') for topic in y_train_2]
        y_train_2 = np.searchsorted(target_names, y_train_2)

        # Compute class weight (1.0 for the biggest class, some factor > 1.0 for smaller classes)
        C = Counter(y_train_2)
        M = max(C.values())
        class_weight = {k: M / v for k, v in C.items()}

        try:
            print('Transform', flush=True)
            x_new_counts = count_vector.fit_transform(data)
            x_train = tfidf_transformer.fit_transform(x_new_counts)

            print('Learning on/off topic', flush=True)
            clf_onTopic = clf_onTopic.fit(x_train, y_train_1)

            print('Learning categories', flush=True)
            clf_categories = clf_categories.fit(x_train, y_train_2)

            classifier_save()

            print('Learned from %d news' % len(data), flush=True)
            learning = Learn.OK
        except Exception as e:
            print(e)
            learning = Learn.NOK


def classifier_filter_news():
    if learning != Learn.OK:
        classifier_load()

    if learning != Learn.OK:
        classifier_learn()

    if learning != Learn.OK:
        print('Can''t filter yet.', flush=True)
        return

    print('Filtering', flush=True)

    stmt = select([News.id, News.tokens]).where(and_(News.publication_id.is_(None), News.auto_cat.is_(None)))
    result = db.engine.execute(stmt)

    try:
        ids, data = zip(*result)
    except ValueError:
        print('Filtered 0 news', flush=True)
        return

    x_new_counts = count_vector.transform(data)
    x_new_tfidf = tfidf_transformer.transform(x_new_counts)
    predicted = clf_onTopic.predict(x_new_tfidf)

    ids = np.compress(np.logical_not(predicted), ids).tolist()  # SQLAlchemy seems to not appreciate numpy arrays

    with dbLock:
        stmt = update(News).where(News.id.in_(ids)).values(auto_cat='OFF TOPIC', category_title='OFF TOPIC')
        db.engine.execute(stmt)
        db.session.commit()

    print('Filtered %d news' % len(data), flush=True)


def classifier_categorize():
    global learning

    if learning != Learn.OK:
        classifier_load()

    if learning != Learn.OK:
        classifier_learn()

    if learning != Learn.OK:
        print('Can''t categorize yet.', flush=True)
        return

    print('Categorizing', flush=True)

    stmt = select([News.id, News.tokens]).where(and_(News.auto_cat.is_(None), News.tokens != ''))
    result = db.engine.execute(stmt)

    try:
        ids, data = zip(*result)
        # print(ids)
        # print(data)
    except ValueError:
        print('Categorized 0 news', flush=True)
        return

    x_new_counts = count_vector.transform(data)
    x_new_tfidf = tfidf_transformer.transform(x_new_counts)

    try:
        predicted = clf_categories.predict(x_new_tfidf)
    except ValueError:  # A problem has appeared, try again after reseting the classifiers
        learning = Learn.NEED
        classifier_learn()
        predicted = clf_categories.predict(x_new_tfidf)

    proba = clf_categories.predict_proba(x_new_tfidf)

    # mappings = [{'id': news_id, 'auto_cat': target_names[cat], 'cat_quote': max(prob)}
    #             for news_id, cat, prob in zip(ids, predicted, proba)]
    mappings = [{'id': news_id, 'auto_cat': target_names[cat], 'cat_quote': max(prob),
                 'category_title': None if max(prob) < 0.85 else target_names[cat]}
                for news_id, cat, prob in zip(ids, predicted, proba)]

    # print(ids)
    # print(predicted)
    # print([max(p) for p in proba])
    # print(mappings)

    with dbLock:
        db.session.bulk_update_mappings(News, mappings)
        db.session.commit()

    print("Categorized %d news" % len(data), flush=True)


# #########################################################"
#
#   News retrieval & processing
#

socket.setdefaulttimeout(10)


def get_page(url, opener):
    if app.debug:
        print(url.encode('utf-8'))

    headers = {'User-agent': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)'}
    # headers = {'User-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:56.0) Gecko/20100101 Firefox/56.0'}
    req = urllib.request.Request(url, None, headers)
    try:
        response = opener.open(req)
    except urllib.error.HTTPError as err:
        flash("Can't read <a href='%s' target='_blank'>%s</a>" % (url, url), "warning")
        print("can't read '%s'." % url.encode('utf-8'))
        return ""
    except socket.timeout:
        flash("Timeout reading <a href='%s' target='_blank'>%s</a>" % (url, url), "warning")
        print("Timeout '%s'." % url.encode('utf-8'))
        return ""
    except UnicodeEncodeError:
        flash("Can't decode <a href='%s' target='_blank'>%s</a>" % (url, url), "warning")
        print("can't decode '%s'." % url.encode('utf-8'))
        return ""

    encoding = response.headers['content-type'].split('charset=')[-1]
    if encoding in ['text/xml', 'text/html', 'application/xml', 'application/rss+xml']:
        encoding = 'utf-8'

    try:
        if response.headers['content-encoding'] == 'gzip':
            gzipFile = gzip.GzipFile(fileobj=response)
            html_source = gzipFile.read().decode(encoding, errors='ignore')
        else:
            html_source = response.read().decode(encoding, errors='ignore')
    except http.client.IncompleteRead as e:
        flash("Partial content for <a href='%s' target='_blank'>%s</a>" % (url, url), "warning")
        print("Partial content for '%s'." % url.encode('utf-8'))
        html_source, = e.args
        html_source = html_source.decode(encoding, errors='ignore')
    url2 = response.geturl()
    response.close()

    return html_source, url2


re_a = re.compile('<a .*?>(.*?)</a>', re.IGNORECASE | re.DOTALL | re.MULTILINE)
re_strong = re.compile('<strong>(.*?)</strong>', re.IGNORECASE | re.DOTALL | re.MULTILINE)
re_null_span = re.compile('<span>(.*?)</span>', re.IGNORECASE | re.DOTALL | re.MULTILINE)
re_iframe = re.compile('<iframe.*?</iframe>', re.IGNORECASE | re.DOTALL | re.MULTILINE)
re_img = re.compile('<img .*?>', re.IGNORECASE | re.DOTALL | re.MULTILINE)
re_comment = re.compile('<!--.*?-->', re.IGNORECASE | re.DOTALL | re.MULTILINE)
re_script = re.compile('<(?:no)?script.*?</(?:no)?script>', re.IGNORECASE | re.DOTALL | re.MULTILINE)
re_style = re.compile('<style.*?</style>', re.IGNORECASE | re.DOTALL | re.MULTILINE)
re_style2 = re.compile(' style\s*=\s*(["\']).*?\\1', re.IGNORECASE | re.DOTALL | re.MULTILINE)
re_link = re.compile('<link.*?>', re.IGNORECASE | re.DOTALL | re.MULTILINE)
re_figure = re.compile('<figure.*?</figure>', re.IGNORECASE | re.DOTALL | re.MULTILINE)
re_blockquote = re.compile('<blockquote.*?</blockquote>', re.IGNORECASE | re.DOTALL | re.MULTILINE)
re_class = re.compile(' class\s*=\s*(["\']).*?\\1', re.IGNORECASE)
re_empty_span = re.compile('<span[^>]*></span>', re.IGNORECASE)
re_empty_p = re.compile('<p> *</p>', re.IGNORECASE)

re_relDate = re.compile('(?P<n>.*?) (?P<scale>days?) ago', re.IGNORECASE)
relDate_scales = {'jours': 'days', 'jour': 'days', 'day': 'days', 'days': 'days'}


def normalize_gdict(gdict, url, source):
    defaults = {'url': '', 'title': '', 'abstract': '', 'text': '', 'date': '', 'time': '',
                'datetime': '', 'datetime2': '', 'dt': None, 'source': ''}

    for k, v in defaults.items():
        if (k not in gdict) or (gdict[k] == None):
            gdict[k] = v

    for _ in range(2):
        gdict['title'] = gdict['title'].replace('<![CDATA[', '').replace(']]>', '')

        # Clean HTML in abstract & text
        gdict['abstract'] = gdict['abstract'].replace('<![CDATA[', '').replace(']]>', '')
        gdict['abstract'] = re_a.sub("\\1", gdict['abstract'])
        gdict['abstract'] = re_strong.sub("\\1", gdict['abstract'])
        gdict['abstract'] = re_null_span.sub("\\1", gdict['abstract'])
        gdict['abstract'] = re_iframe.sub("", gdict['abstract'])
        gdict['abstract'] = re_comment.sub("", gdict['abstract'])
        gdict['abstract'] = re_img.sub("", gdict['abstract'])
        gdict['abstract'] = re_script.sub("", gdict['abstract'])
        gdict['abstract'] = re_style.sub("", gdict['abstract'])
        gdict['abstract'] = re_style2.sub("", gdict['abstract'])
        gdict['abstract'] = re_link.sub("", gdict['abstract'])
        gdict['abstract'] = re_figure.sub("", gdict['abstract'])
        gdict['abstract'] = re_class.sub("", gdict['abstract'])
        gdict['abstract'] = re_style.sub("", gdict['abstract'])
        gdict['abstract'] = re_blockquote.sub("", gdict['abstract'])
        gdict['abstract'] = re_empty_span.sub("", gdict['abstract'])
        gdict['abstract'] = re_empty_p.sub("", gdict['abstract'])
        gdict['text'] = gdict['text'].replace('<![CDATA[', '').replace(']]>', '')
        gdict['text'] = re_a.sub("\\1", gdict['text'])
        gdict['text'] = re_strong.sub("\\1", gdict['text'])
        gdict['text'] = re_null_span.sub("\\1", gdict['text'])
        gdict['text'] = re_iframe.sub("", gdict['text'])
        gdict['text'] = re_comment.sub("", gdict['text'])
        gdict['text'] = re_img.sub("", gdict['text'])
        gdict['text'] = re_script.sub("", gdict['text'])
        gdict['text'] = re_style.sub("", gdict['text'])
        gdict['text'] = re_style2.sub("", gdict['text'])
        gdict['text'] = re_link.sub("", gdict['text'])
        gdict['text'] = re_figure.sub("", gdict['text'])
        gdict['text'] = re_class.sub("", gdict['text'])
        gdict['text'] = re_style.sub("", gdict['text'])
        gdict['text'] = re_blockquote.sub("", gdict['text'])
        gdict['text'] = re_empty_span.sub("", gdict['text'])
        gdict['text'] = re_empty_p.sub("", gdict['text'])

        gdict['text'] = html.unescape(gdict['text'])
        gdict['abstract'] = html.unescape(gdict['abstract'])

    if gdict['abstract'] and '<p>' not in gdict['abstract']:
        gdict['abstract'] = "<p>%s</p>" % gdict['abstract']
    if gdict['text'] and '<p>' not in gdict['text']:
        gdict['text'] = "<p>%s</p>" % gdict['text']

    gdict['url'] = gdict['url'].replace('<![CDATA[', '').replace(']]>', '')
    gdict['url'] = gdict['url'].strip(' \n\t')
    if gdict['url'][:4] != 'http':
        gdict['url'] = urljoin(url, gdict['url'])

    match = re_relDate.match(gdict['date'])
    if match:
        time_dict = {relDate_scales[match['scale']]: float(match['n'])}
        dt = datetime.timedelta(**time_dict)
        gdict['dt'] = datetime.datetime.now() - dt
    else:
        if gdict['datetime'] == '' and gdict['datetime2'] != '':
            gdict['datetime'] = gdict['datetime2']
        if gdict['datetime'] == '':
            gdict['datetime'] = (gdict['date'] + " / " + gdict['time']).strip()
        if gdict['datetime'] != '':
            for datetime_format in source.date_format.split(';;'):
                try:
                    news_date = time.strptime(gdict['datetime'], datetime_format)
                    if news_date.tm_year == 1900:
                        news_date_writable = list(news_date)
                        news_date_writable[0] = datetime.datetime.now().year
                        news_date = time.struct_time(tuple(news_date_writable))
                    gdict['dt'] = datetime.datetime.fromtimestamp(time.mktime(news_date))
                    break
                except ValueError:
                    pass

    if gdict['source'] == '':
        gdict['source'] = source.title
    else:
        gdict['source'] = "%s (%s)" % (source.title, gdict['source'])

    return gdict


# #########################################################"
#
#   Restless API
#


def news_get_many_preprocessor(search_params=None, **kw):
    my_filter = dict(name='publication_id', op='is_null')
    if 'filters' not in search_params:
        search_params['filters'] = []
    search_params['filters'].append(my_filter)


api_manager = APIManager(app, flask_sqlalchemy_db=db)
api_manager.create_api(Categories, methods=['GET', 'POST', 'DELETE', 'PUT'], results_per_page=-1)
api_manager.create_api(Sources, methods=['GET', 'POST', 'DELETE', 'PUT'], results_per_page=-1)
api_manager.create_api(News, methods=['GET', 'POST', 'PUT'], results_per_page=-1,
                       preprocessors=dict(GET_MANY=[news_get_many_preprocessor]))


# #########################################################"
#
#   Flask views
#


@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-cache, no-store'
    return response


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/api/categories/swap/<int:cat_id1>/<int:cat_id2>')
def swap_categories(cat_id1, cat_id2):
    assert cat_id1 != cat_id2

    with dbLock:
        # Swap the sequences of the 2 categories
        stmt = 'UPDATE categories ' + \
               'SET sequence = ((' + \
               '    SELECT SUM(sequence) AS seq_sum ' + \
               '    FROM categories ' + \
               '    WHERE id IN (%d, %d)' % (cat_id1, cat_id2) + \
               ') - sequence) ' + \
               'WHERE id IN (%d, %d);' % (cat_id1, cat_id2)
        db.engine.execute(stmt)
    return ''


@app.route('/api/news/get_news/stream')
def get_news_stream():
    def generate():
        sd = ''
        while True:
            time.sleep(0.5)
            if sd != app.streamData['data']:
                sd = app.streamData['data']
                yield 'data:%s\n\n' % sd
            if app.streamData['close']:
                app.streamData['close'] = False
                break

    return Response(generate(), mimetype='text/event-stream')


def jsonify_flash():
    alerts = []
    messages = get_flashed_messages(with_categories=True)
    for category, message in messages:
        if category == 'error':
            category = 'danger'
        alerts.append({'type': category, 'msg': message})

    return jsonify({'alerts': alerts})


def reject_url(url):
    # TODO: Check with the source parameter if the local url should be rejected for analysis
    return False


@app.route('/api/news/get_news')
def get_news():
    # Setup the cookie jar
    cookies = http.cookiejar.LWPCookieJar()
    handlers = [
        urllib.request.HTTPHandler(),
        urllib.request.HTTPSHandler(),
        urllib.request.HTTPCookieProcessor(cookies)
    ]
    opener = urllib.request.build_opener(*handlers)

    # Get all sources
    stmt = select([Sources]).where(Sources.active == "'True'")
    results = db.engine.execute(stmt)

    all_sources = [s for s in results]
    nb_sources = len(all_sources)
    cur_source = -1
    for source in all_sources:
        if app.debug:
            print(source.title, flush=True)

        # if source.title == 'IE(Internationale de l\'Education)':
        #     cur_source = cur_source

        cur_source += 1
        app.streamData['data'] = str(int(90 * cur_source / nb_sources))

        # Execute login script
        if source.login_script is not None and source.login_script.strip() != '':
            exec(source.login_script, {"__builtins__": None}, {'source': source})

        news2 = []
        for source_url in source.url.split(';;'):
            # Request the news list page
            try:
                html_source, _ = get_page(source_url, opener)
            except ValueError:
                continue
            except Exception as e:
                print(str(e).encode('utf-8'))
                continue

            match_count = 0
            for re_list in source.re_list.split(';;'):
                for match in re.finditer(re_list, html_source, re.DOTALL | re.MULTILINE):
                    match_count += 1
                    gdict = normalize_gdict(match.groupdict(), source_url, source)

                    # Skip some URLs
                    if reject_url(gdict['url']):
                        continue

                    # Skip too old news
                    # if gdict['dt'] and gdict['dt'] < (datetime.datetime.now() - datetime.timedelta(weeks=1)):
                    #     continue

                    # Check if the URL is already in the DB
                    nb_row = get_count(db.session.query(News).filter(News.url == gdict['url']))
                    if nb_row == 0:
                        # Create a news stub
                        news = News(gdict['dt'], gdict['title'], gdict['abstract'],
                                    gdict['text'], gdict['url'], gdict['source'])
                        news2.append(news)

            if match_count == 0:
                flash("No match in source list '%s'" % source.title, "warning")
                print("No match in source list '%s'" % source.title.encode('utf-8'))
                if app.debug:
                    with open('c:\\temp\\source.txt', "w", encoding='utf-8') as text_file:
                        text_file.write(html_source)

        nb_news = len(news2)
        cur_news = -1
        # For each found news
        for news in news2:
            cur_news += 1
            app.streamData['data'] = str(int((90 / nb_sources) * (cur_source + cur_news / nb_news)))

            if source.re_article:
                # Request article page
                try:
                    html2, url2 = get_page(news.url, opener)
                except (ValueError, urllib.error.URLError):
                    continue
                except Exception as e:
                    print(str(e).encode('utf-8'))
                    continue

                if app.debug and url2 != news.url:
                    print("True url '%s'." % url2.encode('utf-8'))
                    news.url = url2

                if html2 and not reject_url(url2):
                    parsed = False
                    # Find article elements
                    for re_article in (r.strip() for r in source.re_article.split(';;')):
                        try:
                            match2 = re.search(re_article, html2, re.DOTALL | re.MULTILINE, timeout=1)
                        except TimeoutError:
                            flash("regex timeout <a href='%s' target='_blank'>%s</a>" % (news.url, news.url), "warning")
                            print("regex timeout '%s'." % news.url.encode('utf-8'))
                            continue

                        if match2 is not None:
                            parsed = True
                            gdict = normalize_gdict(match2.groupdict(), news.url, source)

                            # Check if the news is not already in the database
                            nb_row = get_count(db.session.query(News).filter(News.url == news.url))
                            if nb_row == 0:
                                # Create the article object
                                if gdict['dt'] is not None:
                                    news.date = gdict['dt']
                                if gdict['title']:
                                    news.title = gdict['title']
                                if gdict['abstract']:
                                    news.abstract = gdict['abstract']
                                if gdict['text']:
                                    news.text = gdict['text']

                                # Force default date if None
                                if not news.date:
                                    news.date = datetime.datetime.utcnow()

                                # Post-process the article object
                                if source.post_process:
                                    exec(source.post_process, {"__builtins__": None}, {'news': news, 're': re})

                                # Generate tokens
                                news.tokens = generate_tokens(news)

                                # Save the new article in the DB
                                try:
                                    with dbLock:
                                        db.session.add(news)
                                        db.session.commit()
                                except IntegrityError as e:
                                    with dbLock:
                                        db.session.rollback()
                                    flash("Error with <a href='%s' target='_blank'>%s</a>" % (news.url, news.url),
                                          "warning")
                                    print("Integrity error 1 '%s'." % news.url.encode('utf-8'))
                                    print(str(e).encode('utf-8'))
                                break

                    if not parsed:
                        flash("can't parse <a href='%s' target='_blank'>%s</a>" % (news.url, news.url), "warning")
                        print("can't parse '%s'." % news.url.encode('utf-8'))
                        if app.debug:
                            with open('c:\\temp\\article.txt', "w", encoding='utf-8') as text_file:
                                text_file.write(html2)
            else:
                # Check if the news is not already in the database
                nb_row = get_count(db.session.query(News).filter(News.url == news.url))
                if nb_row == 0:
                    # Post-process the article object
                    if source.post_process:
                        exec(source.post_process, {"__builtins__": None}, {'news': news, 're': re})

                    # Generate tokens
                    news.tokens = generate_tokens(news)

                    # Save the new article in the DB
                    try:
                        with dbLock:
                            db.session.add(news)
                            db.session.commit()
                    except IntegrityError as e:
                        with dbLock:
                            db.session.rollback()
                        flash("Error with <a href='%s' target='_blank'>%s</a>" % (news.url, news.url), "warning")
                        print("Integrity error 2 '%s'." % news.url.encode('utf-8'))
                        print(str(e).encode('utf-8'))

    app.streamData['data'] = '90'

    if learning == Learn.LOAD:
        classifier_load()

    if learning == Learn.NEED:
        classifier_learn()

    app.streamData['data'] = '95'

    classifier_filter_news()
    classifier_categorize()

    app.streamData['data'] = "100"
    app.streamData['close'] = True

    return jsonify_flash()


re_p = re.compile('(<p.*?>.*?</p>)', re.IGNORECASE | re.DOTALL | re.MULTILINE)
re_2p = re.compile('(<p.*?>.*?</p>.*?<p.*?>.*?</p>)', re.IGNORECASE | re.DOTALL | re.MULTILINE)
re_3p = re.compile('(<p.*?>.*?</p>.*?<p.*?>.*?</p>.*?<p.*?>.*?</p>)', re.IGNORECASE | re.DOTALL | re.MULTILINE)
re_4p = re.compile('(<p.*?>.*?</p>.*?<p.*?>.*?</p>.*?<p.*?>.*?</p>.*?<p.*?>.*?</p>)',
                   re.IGNORECASE | re.DOTALL | re.MULTILINE)


@app.route('/gen_review')
def gen_review():
    stmt = select([Categories.sequence, Categories.title.label('cat_title')]) \
        .where(Categories.id >= 0) \
        .order_by(Categories.sequence)
    categories = db.engine.execute(stmt)

    html = ""
    for cat in categories:
        html += "<h1 class='Heading-1'>%s</h1>\n" % cat.cat_title

        stmt2 = select([News.id, News.source_title, News.date, News.title, News.abstract, News.text, News.url]) \
            .where(and_(News.category_title == cat.cat_title,
                        News.publication_id.is_(None))) \
            .order_by(News.date)
        results = db.engine.execute(stmt2)

        for row in results:
            html += "<h2 class='Heading-2'>%s (%s)</h2>\n" % (row.title, row.source_title)
            html += "<p class='DateTime'>%s</p>\n" % (row.date and row.date.strftime("%d/%m/%Y @ %H:%M"))
            base_text = row.abstract + "\n" if row.abstract != '' else ''
            text2 = base_text + row.text
            for re_np in [re_p, re_2p, re_3p, re_4p]:
                match = re_np.search(row.text)
                if match is not None:
                    text2 = base_text + match.group(1)
                    if len(text2) < 500:
                        continue
                break
            html += text2 + "\n"
            if row.url[:4] == 'http':
                html += "<p class='Hyperlink'><a href='%s' target='_blank'>%s</a></p>\n" % (row.url, row.url)

    html = html.replace("<p>", "<p class='Normal'>")

    review = {'date': datetime.datetime.now(), 'html': html}

    db.session.commit()

    return jsonify(**review)


@app.route('/accept_review')
def accept_review():
    with dbLock:
        stmt = insert(Publications).values(date=datetime.datetime.now())
        result = db.engine.execute(stmt)
        pub_id = result.inserted_primary_key[0]

        stmt = update(News).values(publication_id=pub_id).where(and_(News.category_title.isnot(None),
                                                                     News.publication_id.is_(None)))
        db.engine.execute(stmt)
        db.session.commit()

    # learning = Learn.NEED
    # classifier_learn()
    return ''


@app.route('/revert_review')
def revert_review():
    global learning

    with dbLock:
        pub_id = db.session.query(func.max(News.publication_id)).scalar()

        stmt = update(News).values(publication_id=None).where(News.publication_id == pub_id)
        db.engine.execute(stmt)
        db.session.commit()

    # learning = Learn.NEED
    return ''


def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


@app.route('/shutdown')
def shutdown():
    db.session.commit()
    shutdown_server()
    return 'Server shutting down...'


# #########################################################
#
#   PressRrview main
#


if __name__ == '__main__':
    os.system('mode con cols=160')
    print(sys.version)
    print('Open the firewall with:')
    print('netsh firewall add portopening TCP %d "MyDGNews"' % PORT)
    print('netsh advfirewall firewall add rule name="MyDGNews TCP Port %d" dir=in action=allow protocol=TCP '
          'localport=%d' % (PORT, PORT))
    print('netsh advfirewall firewall add rule name="MyDGNews TCP Port %d" dir=out action=allow protocol=TCP '
          'localport=%d' % (PORT, PORT))

    app.run(host='0.0.0.0', port=PORT, debug=True, threaded=True)
    # app.run(host='0.0.0.0', port=PORT, threaded=True)
