import os
import re
import json
import abc

from pygments import lexers

from lm_eval.base import rf, PerplexityTask
from lm_eval.utils import sh
from best_download import download_file


# overlapped_repos.json
OVERLAP = ["xtuJSer/CoCoMusic", "nodejs/node-convergence-archive", "WhiteBlue/bilibili-sdk-go", "alex-klepa/rails4-bootstrap-devise-cancan-omniauth", "ellson/MOTHBALLED-graphviz", "mujx/nheko", "g4zhuj/grpc-wrapper", "yfgeek/BlockVotes", "Maks3w/FR3DLdapBundle", "geokit/geokit-rails3", "codecasts/spa-starter-kit", "andrewjstone/rabble", "ybogdanov/node-sync", "konlpy/konlpy", "TianzhongSong/Real-Time-Action-Recognition", "raggi/async_sinatra", "petermattis/goid", "xuningjack/CustomClock", "V-E-O/PoC", "eugeneek/SmileBar", "LeechanX/Netflix-Recommender-with-Spark", "vinzenz/libpypa", "MaxBelkov/visualsyslog", "applicationsonline/librarian", "bobappleyard/golisp", "codeskyblue/gobuild", "atomix/copycat", "hss01248/PhotoOut", "scateu/PyWGS84ToGCJ02", "White-Oak/qml-rust", "stevepapa/veria", "jamie-allen/effective_akka", "yorkie/rust.js", "colintheshots/AnDevCon-RxPatterns", "smuyyh/IncrementallyUpdate", "pwaller/goupx", "dodola/ToyView", "philipbjorge/Infinite-Social-Wall", "cube-js/cube.js", "jschwindt/rjcrop", "levex/osdev", "tedsta/deeplearn-rs", "xdavidhu/probeSniffer", "MikeStall/DataTable", "yawaramin/scala-modules", "itmad/Tpay_Svr", "AliasIO/Swiftlet", "MrNerverDie/MiniWeChat-Client", "elvanja/jenkins-gitlab-hook-plugin", "microsoft/Cognitive-Face-Android", "wangjiegulu/androidInject", "bosko/rmre", "manuelbernhardt/reactive-web-applications", "b4winckler/vim", "scoopr/vectorial", "Qihoo360/zeppelin", "nolanlawson/CustomFastScrollViewDemo", "frankmcsherry/COST", "hjr3/weldr", "TheKhaeng/pushdown-anim-click", "manub/scalatest-embedded-kafka", "huchenme/github-trending-api", "cdflynn/touchdemo", "jserv/xv6-x86_64", "ganemone/NodeRequirer", "wywu/LAB", "mmin18/LayoutCast", "dodola/WeexOne", "openstacknetsdk/openstack.net", "hortonworks/kubernetes-yarn", "alienfast/elastic-beanstalk", "pablosmedina/ckite", "mexx/FeatureSwitcher", "jndok/harpoon", "wookayin/tensorflow-talk-debugging", "Megvii-CSG/MegReader", "ant0ine/go-urlrouter", "DongjunLee/conversation-tensorflow", "antham/chyle", "zsiciarz/24daysofrust", "chinchang/cta.js", "david-gpu/srez", "dcodeIO/MetaScript", "fanfoudroid/fanfoudroid", "servergrove/KnowledgeBase", "PostHog/posthog", "jscottsmith/react-scroll-parallax", "ThomasBurleson/angularjs-logDecorator", "atlasr-org/atlasr", "fogleman/pt", "tunitowen/DevDrawer", "uecode/qpush-bundle", "mitchmindtree/elmesque", "mixandjam/Celeste-Movement", "mkottman/AndroLua", "easychen/LazyAudioBook", "Onskreen/cornerstone", "kohana/database", "jdegoes/zio-workshop", "karmi/pushr", "dcaoyuan/spray-socketio", "totond/TextPathView", "ohnosequences/sbt-s3-resolver", "elodina/scala-kafka", "Verizon/funnel", "axel22/Ctries", "apg/wipes", "nytimes/marvin", "LearningOS/rcore_step_by_step", "Sottti/OkHttp-Volley-Gson", "Kode/Kha", "Gridstone/RxStore", "dojo/intern-only-dojo", "llSourcell/3D_Pose_Estimation", "Tencent/TscanCode", "EvilCult/Video-Downloader", "a312863063/seeprettyface-generator-wanghong", "Polymer/shop", "nolanlawson/blob-util", "apache/incubator-retired-gearpump", "rapidloop/rtop-bot", "tenderlove/tusk", "skellock/typescript-with-electron-react-kit", "ashqal/NightOwl", "hsiaosiyuan0/naive", "ivanvanderbyl/cloudist", "re54k/mobileweb-utilities", "yatish27/linkedin-scraper", "spring-projects/spring-scala", "DavyJonesLocker/party_foul", "sos-os/kernel", "darkskyapp/forecast-ruby", "sfturing/hosp_order", "wukezhan/air", "imotov/elasticsearch-analysis-morphology", "makaroni4/sandi_meter", "tailhook/rust-argparse", "zeisler/active_mocker", "SoftEtherVPN/Win10Pcap", "axle-h/Retro.Net", "aicaprio/CurtainView", "playframework/play-scala-websocket-example", "JlUgia/beauty-treatment-android-animations", "siddontang/moonmq", "c9s/GetOptionKit", "ianpreston/redditfs", "ennorehling/dlmalloc", "shelfio/aws-lambda-libreoffice", "DhavalKapil/icmptunnel", "ndt-project/ndt", "n3-charts/line-chart", "facebook/relay", "fengzhizi715/SAF-AOP", "vandernorth/NecroBot.GUI", "amplab/MLI", "dddExperiments/SFMToolkit", "tito/2048", "ValveSoftware/steamos_kernel", "alexcrichton/wasm-gc", "jimthunderbird/php-to-c-extension", "adrien2p/nestjs-graphql", "Manu343726/Turbo", "NZKoz/rails_xss", "SullyChen/Chai", "fwbrasil/activate", "Noctem/Monocle", "Brain-WP/Cortex", "khanrc/pt.darts", "srkirkland/DataAnnotationsExtensions", "yaauie/redis-copy", "MengLiPKU/VideoStitch", "knutwalker/typed-actors", "yeungeek/monkey-android", "ervandew/eclim", "Tasssadar/MultiROMMgr", "toidiu/learn-rust", "AArnott/ImmutableObjectGraph", "petereigenschink/steganography.js", "marquisXuan/netty", "editorconfig/editorconfig-visualstudio", "nkallen/querulous", "bernhard2202/improved-video-gan", "ProgrammationAndroid/Laravel-Passport-Android", "tdewolff/push", "AndreyAkinshin/knockout-mvc", "mariotaku/RefreshNow", "halfkiss/ZjDroid", "facebookarchive/FBMock", "vmg/crustache", "yonch/fastpass", "jamesmanning/RunProcessAsTask", "amplab/keystone", "lidong1665/Android-ble", "burke/matcher", "fregu856/deeplabv3", "spray/spray-template", "bilibili/jni4android", "joel16/PSV-VSH-Menu", "laravelbook/laravel4-phpstorm-helper", "sparkfun/SparkFun_MPU-9250-DMP_Arduino_Library", "galnir/Master-Bot", "StreakYC/StreakSecureGmail", "ipfs-shipyard/java-ipfs-http-client", "fortrabbit/slimcontroller", "Gnaf/GenesisBlockZero", "iambus/youku-lixian", "codemy/shopper", "databricks/spark-corenlp", "ManuelPeinado/FadingActionBar", "PHPOffice/PhpProject", "fabian7593/MagicalCamera", "zachlatta/sshtron", "KyleU/databaseflow", "momo5502/cod-exploits", "mkuthan/example-spark-kafka", "espressif/ESP8266_AT", "apex/apex-go", "guregu/kami", "echen/link-prediction", "Barnacules/Codegasm", "andacata/HybridIgniter", "Coderockr/orcamentos", "ardanlabs/kit", "mDialog/smoke", "bsadeh/scalastic", "lampo1024/TsBlog", "fishioon/douyu", "blangel/wrk", "akretion/ooor", "axemclion/grunt-saucelabs", "pagarme/teleport", "dolittle/Bifrost", "tburry/pquery", "htm-community/flink-htm", "dalingge/GankGirl", "lincanbin/Holy-Lance", "StrawberryFlavor/Selenium-Framework", "Versent/redux-crud", "egeloen/ivory-google-map", "aloiscochard/scato", "reactphp-legacy/socket-client", "xwjie/MyRestUtil", "ochrons/scalajs-spa-tutorial", "nyddle/pystash", "Propaganistas/Laravel-Intl", "bsspirit/maven_hadoop_template", "databricks/spark-training", "nullpomino/nullpomino", "joehewitt/nerve", "orangeduck/LuaAutoC", "ArsenArsen/KShare", "REMath/implementations", "sharpdx/SharpDX-Samples", "fluffyemily/cross-platform-rust", "RobertYim/ShadowsocksX", "kliment/Sprinter", "googleprojectzero/Street-Party", "willcrichton/lia", "scalawarrior/scalawarrior", "stevej/scala-json", "joereynolds/mort", "bazad/blanket", "theburningmonk/SimpleSpeedTester", "WesleyAC/plotty-bird", "postrank-labs/goliath", "scala-records/scala-records", "tjoudeh/JWTAspNetWebApi", "meteorhacks/flow-components", "m4b/dryad", "OneOfOne/lfchan", "ajbrock/SMASH", "ssloy/tinyraycaster", "bradfitz/latlong", "sherlockchou86/ZhiHuDaily.UWP", "Scalingo/go-graceful-restart-example", "RunningGump/gsxt_captcha", "oxoooo/mr-mantou-android", "Fakerino/Fakerino", "championswimmer/vuex-persist", "angular-university/angular-firebase-app", "timwis/csv-schema", "cnnblike/discord-lite", "aheadley/homeworld", "jhades/angular2-redux-store", "AlbertGrobas/PolygonImageView", "NetEase/lordofpomelo", "google/fancy-regex", "mdr/ascii-graphs", "zalando/grafter", "PatrickJS/angular-md5", "sr/dwm", "novoda/notils", "seletskiy/hastur", "rjagerman/glint", "LuckyJayce/ViewPagerIndicator", "incredibleindishell/sqlite-lab", "cliftonm/FlowSharp", "palvaro/molly", "ezekg/xo", "tangtanglove/blockscloud", "scala-blitz/scala-blitz", "redbear/Duo", "SmartXiaoMing001/Chinese-Cipher-Of-SM2-SM3-SM4", "web3j/web3j-spring-boot-starter", "zhitaocai/CocosCreator-Multi-resolution-Adapter", "tiagonmas/Windows-Loopback-Exemption-Manager", "microsoft/LQ-Nets", "commonsguy/cwac-anddown", "sakeven/httpproxy", "totemstech/neuraln", "clemens/later_dude", "mozilla/telemetry-airflow", "Stratio/crossdata", "nmosafi/aspComet", "SimplyBuilt/SimonSays", "DmitryMalkovich/material-design-dimens", "msurguy/laravel-shop-menu", "garethr/kubetest", "marek-stoj/NReadability", "briangonzalez/rgbaster.js", "GravityLabs/HPaste", "bahlo/goat", "maysrp/webdir", "NitrogenEmulator/Nitrogen", "philwantsfish/shard", "mattt/rack-push-notification", "starburst1977/out-of-words", "sbt/sbt-start-script", "rails/routing_concerns", "zint/zint", "developerforce/Force.com-JavaScript-REST-Toolkit", "zgzczzw/ZHFollowButton", "grimfang4/sfxr", "silentsignal/sheep-wolf", "dwrensha/gj", "addyosmani/largescale-demo", "dsplaisted/PCLStorage", "futuresimple/broadcast", "redhotvengeance/deep_thought", "tum-vision/tum_ardrone", "Mparaiso/Silex-Blog-App", "sourcelair/ceryx", "velvia/ScalaStorm", "ibm-watson-iot/blockchain-samples", "tpys/face-everthing", "RichardLitt/awesome-conferences", "vaquarkhan/Apache-Kafka-poc-and-notes", "stackia/XIME", "NotifyMeHQ/notifyme", "scotch/engineauth", "puppetlabs/puppet-module-tool", "smartyuge/SuperIndicator", "floveluy/Burnjs", "ragunathjawahar/instant-adapter", "trochette/Angular-Design-Patterns-Best-Practices", "chsakell/spa-webapi-angularjs", "quentinhardy/scriptsAndExploits", "dotboris/eldritch", "risq/investigator", "johnmq/john", "yingDev/rxrs", "danhper/structomap", "bchanx/slidr", "lezhnev74/openapi-psr7-validator", "ajwhite/render-if", "aspnet/FileSystem", "kelseyhightower/kargo", "chandu0101/sri", "goetas/xsd2php", "gcorne/wp-react-boilerplate", "postgrespro/imgsmlr", "0x09AL/go-deliver", "debasishg/cqrs-akka", "RJ/playdar-core", "christinang89/wfh-ninja", "netxfly/Transparent-Proxy-Scanner", "ksvc/MediaParser", "risuiowa/rails-jquery-autocomplete", "tucano/UnityRandom", "maxim/skeptick", "dizda/CloudBackupBundle", "codekansas/keras-language-modeling", "DozerMapper/dozer", "tulleuchen/jirastopwatch", "Hopetree/E-commerce-crawlers", "OSAS/strapping-mediawiki", "grosser/smusher", "hussachai/play-scalajs-showcase"]

class Code(PerplexityTask, abc.ABC):
    VERSION = 0
    LANG_NAME = None

    def download(self):
        if not os.path.exists('evaldata/Code-sampled100'):
            os.makedirs("evaldata/Code-sampled100", exist_ok=True)
            download_file("https://zenodo.org/record/6363556/files/unseen_test_sets.tar.gz?download=1", "evaldata/unseen_test_sets.tar.gz")
            sh("cd evaldata/ && tar -zxf unseen_test_sets.tar.gz")

        self.lexer = lexers.get_lexer_by_name(self.LANG_NAME.lower())
        self.overlapped = set(OVERLAP)

    def fewshot_description(self):
        # TODO: figure out fewshot description
        return ""

    def has_validation_docs(self):
        return True

    def has_train_docs(self):
        return False

    def has_test_docs(self):
        return False
    
    def validation_docs(self):
        for root, _, files in os.walk('evaldata/Code-sampled100/{}'.format(self.LANG_NAME)):
            for file in files:
                content = open(os.path.join(root, file)).read().strip()
                if content:
                    yield open(os.path.join(root, file)).read()
                
    def train_docs(self):
        pass

    def test_docs(self):
        pass

    def doc_to_target(self, doc):
        return doc
    
    def count_words(self, doc):
        # count number of words in *original doc before detokenization*
        return len(list(self.lexer.get_tokens(doc)))

class CodePython(Code):
    LANG_NAME = "Python"

class CodeCpp(Code):
    LANG_NAME = "C++"

class CodeC(Code):
    LANG_NAME = "C"

class CodeCS(Code):
    LANG_NAME = "C#"

class CodeRuby(Code):
    LANG_NAME = "Ruby"

class CodeRust(Code):
    LANG_NAME = "Rust"

class CodeJava(Code):
    LANG_NAME = "Java"

class CodeJavaScript(Code):
    LANG_NAME = "JavaScript"

class CodeTypeScript(Code):
    LANG_NAME = "TypeScript"

class CodeGo(Code):
    LANG_NAME = "Go"

class CodeScala(Code):
    LANG_NAME = "Scala"

class CodePHP(Code):
    LANG_NAME = "PHP"
