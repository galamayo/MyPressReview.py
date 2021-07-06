/**
 * Created by Thierry on 18/11/2014.
 */
"use strict";

var app = angular.module("MyDGNews", ['ui.bootstrap'], function ($compileProvider) {
  $compileProvider.aHrefSanitizationWhitelist(/^\s*(https?|ftp|mailto|file|javascript):/);
});

app.directive('draggable', function() {
    return function(scope, element) {
        // this gives us the native JS object
        var el = element[0];

        el.draggable = true;

        el.addEventListener(
            'dragstart',
            function(e) {
                e.dataTransfer.effectAllowed = 'move';
                e.dataTransfer.setData('Text', this.id);
                this.classList.add('drag');
                return false;
            },
            false
        );

        el.addEventListener(
            'dragend',
            function(e) {
                this.classList.remove('drag');
                return false;
            },
            false
        );
    }
});

app.directive('droppable', function($parse) {
    return {
//        scope: {
//            drop: '&'
//        },
        link: function(scope, element, attrs) {
            // again we need the native object
            var el = element[0];

            el.addEventListener(
                'dragover',
                function(e) {
                    e.dataTransfer.dropEffect = 'move';
                    // allows us to drop
                    if (e.preventDefault) e.preventDefault();
                    this.classList.add('over');
                    return false;
                },
                false
            );

            el.addEventListener(
                'dragenter',
                function(e) {
                    this.classList.add('over');
                    return false;
                },
                false
            );

            el.addEventListener(
                'dragleave',
                function(e) {
                    this.classList.remove('over');
                    return false;
                },
                false
            );

            el.addEventListener(
                'drop',
                function(e) {
                    // Stops some browsers from redirecting.
                    if (e.stopPropagation) e.stopPropagation();
                    if (e.preventDefault) e.preventDefault();

                    this.classList.remove('over');

                    var binId = this.id;
                    var item = document.getElementById(e.dataTransfer.getData('Text'));

                    // call the passed drop function
                    scope.$apply(function() {
//                        var fn = scope.drop();
                        var fn = $parse(scope.app[attrs.drop]);
                        if ('undefined' !== typeof fn) {
                            fn(item.id, binId);
                        }
                    });

                    return false;
                },
                false
            );
        }
    }
});

app.controller("MyDGNewsCtrl", ['$scope', '$http', '$sce', function ($scope, $http, $sce) {
    var app = this;

    /* */

    app.curTab = "home";
    app.grabStatus = 100;
    app.alerts_news = [];
    app.alerts_emm = [];
    app.oneNews = { date: roundDateTime(), title: '', abstract: '', text: '', url: '', source_title: ''};
    app.emm = '';
    app.review = { date: null, html: ''};

    /* date picker */

    function roundDateTime() {
        var now = new Date();

        now.setMilliseconds(0);
        now.setSeconds(0);
        now.setMinutes(Math.round(now.getMinutes() / 15) * 15);

        return now;
    }

    app.openPicker = function($event) {
        $event.preventDefault();
        $event.stopPropagation();

        app.openedPicker = true;
    };

    app.dateOptions = {
        formatYear: 'yyyy',
        startingDay: 1
    };

    /* Manage categories */

    app.refreshCategories = function() {
        $http.get("/api/categories").success(function(data) {
            app.categories = data.objects;
        });
    };

    app.filterCategories = function(category) {
        return category.id > 0;
    };

    app.addCategory = function() {
        var seq = 0;
        for (var i = 0; i < app.categories.length; i++) {
            if (seq <= app.categories[i].sequence) {
                seq = app.categories[i].sequence + 1;
            }
        }
        $http.post("/api/categories", {"sequence": seq, "title": "New category"})
            .success(function(data) {
                app.categories.push(data);
            });
    };

    app.updateCategory = function(category) {
        $http.put("/api/categories/" + category.id, category);
    };

    function swapCategories(cat1, cat2) {
        $http.get("/api/categories/swap/" + cat1.id + "/" + cat2.id).success(function() {
            var tmp = cat1.sequence;
            cat1.sequence = cat2.sequence;
            cat2.sequence = tmp;
        });
    }

    app.upCategory = function(category) {
        var cat2 = null;
        for (var i = 0; i < app.categories.length; i++) {
            if (app.categories[i].sequence < category.sequence && (cat2 === null || app.categories[i].sequence > cat2.sequence)) {
                cat2 = app.categories[i];
            }
        }

        swapCategories(category, cat2);
    };

    app.downCategory = function(category) {
        var cat2 = null;
        for (var i = 0; i < app.categories.length; i++) {
            if (app.categories[i].sequence > category.sequence && (cat2 === null || app.categories[i].sequence < cat2.sequence)) {
                cat2 = app.categories[i];
            }
        }

        swapCategories(category, cat2);
    };

    app.delCategory = function(category) {
        $http.delete("/api/categories/" + category.id)
            .success(function() {
                app.categories.splice(app.categories.indexOf(category), 1);
            });
    };


    /* Manage Sources */

    app.refreshSources = function() {
        $http.get("/api/sources").success(function(data) {
            app.sources = data.objects;
        });
    };

    app.addSource = function() {
        $http.post("/api/sources", {"title": "New source"})
            .success(function(data) {
                app.sources.push(data);
            });
    };

    app.updateSource = function(source) {
        $http.put("/api/sources/" + source.id, source);
    };

    app.delSource = function(source) {
        $http.delete("/api/sources/" + source.id).success(function() {
            app.sources.splice(app.sources.indexOf(source), 1);
        });
    };

    /* Sort news */

    app.refreshNews = function() {
        $http.get("/api/news").success(function(data) {
            app.news = data.objects;
        });
    };

    app.grabNews = function() {
        var source = new EventSource('/api/news/get_news/stream');
        source.onmessage = function (event) {
            app.grabStatus = event.data;
            $scope.$digest()
        };

        $http.get("/api/news/get_news").success(function(data) {
            source.close();
            app.grabStatus = 100;
            app.alerts_news = data.alerts;

            $http.get("/api/news").success(function(data) {
                app.news = data.objects;
            });
        });
    };

    app.closeAlert = function(alerts, index) {
        alerts.splice(index, 1);
    };
    
    app.addHiddenNews = function(alerts, index, url) {
        $http.post("/api/news", {"date": roundDateTime(), "title": 'dummy', "abstract": '', "text": '', "url": url,
                                 "source_title": 'dummy'})
            .success(function() {
                alerts.splice(index, 1);
            });
    };

    app.delNews = function(news) {
        news.category = -1;
        $http.put("/api/news/" + news.id, news);
    };

    app.showNews = function(news) {
         alert(news.abstract + '\n\n' + news.text);
    };

    function find_news_index(news_id) {
        for (var i = 0; i < app.news.length; i++) {
            if (app.news[i].id == news_id) {
                return i;
            }
        }
        return -1;
    }

    app.renderHtml = function(html_code) {
        return $sce.trustAsHtml(html_code);
    };

    function get_category_title(category_id) {
        for (var i = 0; i < app.categories.length; i++) {
            if (app.categories[i].id == category_id) {
                return app.categories[i].title;
            }
        }
        return null;
    }

    var percentColors = [
        { pct: 0.0, color: { r: 255, g: 0x00, b: 0 } },
        { pct: 0.5, color: { r: 255, g: 255, b: 0 } },
        { pct: 1.0, color: { r: 0, g: 255, b: 0 } } ];

    app.getColorForPercentage = function(pct) {
        for (var i = 1; i < percentColors.length-1; i++) {
            if (pct < percentColors[i].pct) {
                break;
            }
        }

        var lower = percentColors[i - 1],
            upper = percentColors[i],
            rangePct = (pct - lower.pct) / (upper.pct - lower.pct),
            color = {
                r: Math.floor(lower.color.r * (1 - rangePct) + upper.color.r * rangePct),
                g: Math.floor(lower.color.g * (1 - rangePct) + upper.color.g * rangePct),
                b: Math.floor(lower.color.b * (1 - rangePct) + upper.color.b * rangePct)
            };
        return 'rgb(' + [color.r, color.g, color.b].join(',') + ')';
    };

    function eid2id(eid) {
        return parseInt(eid.replace(/[^-0-9]/g,''));
    };

    app.newsListDrop = function(item, bin) {
        var news_id = eid2id(item);

        $http.put("/api/news/" + news_id, {category_title: null})
            .success(function(data) {
                app.news[find_news_index(news_id)] = data;
            });
    };

    app.newsCategoryDrop = function(item, bin) {
        var news_id = eid2id(item),
            category_id = eid2id(bin);

        $http.put("/api/news/" + news_id, {category_title: get_category_title(category_id)})
            .success(function(data) {
                app.news[find_news_index(news_id)] = data;
            });
    };

    /* Review */

    app.genReview = function() {
        $http.get("/gen_review").success(function(data) {
            app.review = data;
            setTimeout(function() { SelectContent('review'); }, 100);
        });
    };

    app.acceptReview = function() {
      $http.get("/accept_review").success(function(data) {
        $http.get("/api/news").success(function(data) {
            app.news = data.objects;
        });
      })
    };

    app.revertReview = function() {
      $http.get("/revert_review").success(function(data) {
        $http.get("/api/news").success(function(data) {
            app.news = data.objects;
        });
      })
    };

    app.refreshCategories();
    app.refreshSources();
    app.refreshNews();

}]);
