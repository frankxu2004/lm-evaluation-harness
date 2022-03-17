import imp
import os
import re
import json
import abc

from pygments import lexers

from lm_eval.base import rf, PerplexityTask
from lm_eval.utils import sh
from best_download import download_file

class Code(PerplexityTask, abc.ABC):
    VERSION = 0
    LANG_NAME = None

    def download(self):
        if not os.path.exists('evaldata/Code-sampled100'):
            os.makedirs("evaldata/Code-sampled100", exist_ok=True)
            download_file("https://zenodo.org/record/6363556/files/unseen_test_sets.tar.gz?download=1", "evaldata/unseen_test_sets.tar.gz")
            sh("cd evaldata/ && tar -zxf unseen_test_sets.tar.gz")

        self.lexer = lexers.get_lexer_by_name(self.LANG_NAME.lower())
        self.overlapped = set(json.load(open('/home/fangzhex/md/projects/lm-evaluation-harness/overlapped_repos.json')))

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
