# https://dl.acm.org/doi/pdf/10.1145/3319502.3374784
# https://raw.githubusercontent.com/MattiaRacca/tuning_al_gui/master/src/range_al/range_al.py
#!/usr/bin/env python

from __future__ import division
import numpy as np
from scipy.special import expit
from scipy.stats import beta
from copy import copy
from enum import Enum
from time import time
from abc import abstractmethod, ABCMeta

# compatible with Python 2 *and* 3:
ABC = ABCMeta('ABC', (object, ), {'__slots__': ()})


class Learner(ABC):
    def __init__(self,
                 domain,
                 value_distribution,
                 safe=False,
                 safe_phi=0.0,
                 profiling=False):
        self.domain = domain
        self.last_query = None
        self.last_estimate = None
        self.last_answer = None
        self.query_history = []
        self.estimate_history = []
        self.answer_history = []
        self.value_distribution = copy(value_distribution)
        self.safe = safe
        self.safe_phi = safe_phi
        self.profiling = profiling

    @property
    def value_distribution(self):
        return self._value_distribution

    @value_distribution.setter
    def value_distribution(self, value_distribution):
        try:
            is_true_categorical_pdf(value_distribution)
        except AssertionError:
            raise LearnerException(0)
        self._value_distribution = value_distribution

    @property
    def safe_phi(self):
        return self._safe_phi

    @safe_phi.setter
    def safe_phi(self, safe_phi):
        if self.safe:
            self._safe_phi = safe_phi
            self.tau_cutoff = compute_distribution_tail_cutoff(
                self.value_distribution, self._safe_phi)

    @property
    def last_query(self):
        return self._last_query

    @last_query.setter
    def last_query(self, last_query):
        self._last_query = last_query
        if last_query is not None:
            self.query_history.append(last_query)

    @property
    def last_estimate(self):
        return self._last_estimate

    @last_estimate.setter
    def last_estimate(self, last_estimate):
        self._last_estimate = last_estimate
        if last_estimate is not None:
            self.estimate_history.append(last_estimate)

    @property
    def last_answer(self):
        return self._last_answer

    @last_answer.setter
    def last_answer(self, last_answer):
        self._last_answer = last_answer
        if last_answer is not None:
            self.answer_history.append(last_answer)

    @abstractmethod
    def choose_query(self):
        pass

    @abstractmethod
    def update_model(self, answer, query=None):
        pass

    @abstractmethod
    def reset_learner(self):
        self.last_query = None
        self.last_estimate = None
        self.last_answer = None
        self.query_history = []
        self.estimate_history = []
        self.answer_history = []

    @abstractmethod
    def get_current_estimate(self):
        pass

    def n_query_learn(self, query_number, true_min, true_max):
        choose_query_results = {}
        update_model_results = {}
        estimates = {}
        for i in range(0, query_number):
            # ask
            choose_query_results[i] = self.choose_query()
            # get answer
            if self.last_query < true_min:
                answer = LearnerAnswers.HIGHER
            elif true_min <= self.last_query <= true_max:
                answer = LearnerAnswers.FINE
            else:
                answer = LearnerAnswers.LOWER
            # update
            update_model_results[i] = self.update_model(answer)
            estimates[i] = self.get_current_estimate()
        return choose_query_results, update_model_results, estimates

    def enforce_strict_range(self, strict_query_range):
        # ensures no question is asked outside the strict_query_range
        # ONLY WORKING for learners with a safe_phi

        new_distribution = copy(self.value_distribution)
        for index, value in enumerate(self.domain):
            if not (strict_query_range[0] <= value <= strict_query_range[1]):
                new_distribution[index] = 0.0
        new_distribution /= np.sum(new_distribution)
        self.value_distribution = new_distribution
        self.tau_cutoff = compute_distribution_tail_cutoff(
            self.value_distribution, self._safe_phi)

    def __copy__(self):
        copied_learner = type(self)(domain=copy(self.domain),
                                    value_distribution=copy(
                                        self.value_distribution))
        copied_learner.__dict__.update(self.__dict__)
        copied_learner.reset_learner()
        return copied_learner


class MinMaxLearner(Learner):
    def __init__(self,
                 n_evidence_minmax,
                 logistic_k_minmax=50.0,
                 *args,
                 **kwargs):
        self.logistic_k_minmax = logistic_k_minmax
        self.n_evidence_minmax = n_evidence_minmax
        super(MinMaxLearner, self).__init__(*args, **kwargs)

    @Learner.value_distribution.setter
    def value_distribution(self, value_distribution):
        super(MinMaxLearner,
              type(self)).value_distribution.fset(self, value_distribution)
        # (re)compute related distributions
        self.reset_learner()

    def reset_learner(self):
        # (re)compute related distributions
        try:
            super(MinMaxLearner, self).reset_learner()
            self.max_distribution = compute_sample_max_distribution(
                self.value_distribution, self.n_evidence_minmax)
            self.min_distribution = compute_sample_min_distribution(
                self.value_distribution, self.n_evidence_minmax)
            # add initial estimate (before any query is made) to the estimate history
            self.last_estimate = self.get_current_estimate()
        except ValueError as error:
            raise error

    def update_model(self, answer, query=None):
        timer = CodeTimer()
        with timer:
            q = self.last_query if query is None else query

            if q is None:
                raise LearnerException(2)

            if self.profiling:
                n = self.domain.shape[0]
                max_filter_test = np.ones(n) / n
                min_filter_test = np.ones(n) / n

            if answer == LearnerAnswers.FINE:
                self.max_distribution = simple_query_filter(
                    self.domain, self.max_distribution, q,
                    self.logistic_k_minmax, True)
                self.min_distribution = simple_query_filter(
                    self.domain, self.min_distribution, q,
                    self.logistic_k_minmax, False)
                if self.profiling:
                    max_filter_test = simple_query_filter(
                        self.domain, max_filter_test, q,
                        self.logistic_k_minmax, True)
                    min_filter_test = simple_query_filter(
                        self.domain, min_filter_test, q,
                        self.logistic_k_minmax, False)
            elif answer == LearnerAnswers.HIGHER:
                self.max_distribution = simple_query_filter(
                    self.domain, self.max_distribution, q,
                    self.logistic_k_minmax, True)
                self.min_distribution = simple_query_filter(
                    self.domain, self.min_distribution, q,
                    self.logistic_k_minmax, True)
                if self.profiling:
                    max_filter_test = simple_query_filter(
                        self.domain, max_filter_test, q,
                        self.logistic_k_minmax, True)
                    min_filter_test = simple_query_filter(
                        self.domain, min_filter_test, q,
                        self.logistic_k_minmax, True)
            elif answer == LearnerAnswers.LOWER:
                self.max_distribution = simple_query_filter(
                    self.domain, self.max_distribution, q,
                    self.logistic_k_minmax, False)
                self.min_distribution = simple_query_filter(
                    self.domain, self.min_distribution, q,
                    self.logistic_k_minmax, False)
                if self.profiling:
                    max_filter_test = simple_query_filter(
                        self.domain, max_filter_test, q,
                        self.logistic_k_minmax, False)
                    min_filter_test = simple_query_filter(
                        self.domain, min_filter_test, q,
                        self.logistic_k_minmax, False)
            else:
                raise LearnerException(1)

            self.last_answer = answer
            self.last_query = None
            self.last_estimate = self.get_current_estimate()

        if not self.profiling:
            return True
        else:
            update_results = {
                'max_distribution': self.max_distribution,
                'min_distribution': self.min_distribution,
                'max_filter_test': max_filter_test,
                'min_filter_test': min_filter_test,
                'time': timer.elapsed
            }
            return update_results

    def get_current_estimate(self):
        min_estimate = self.domain[np.argmax(self.min_distribution)]
        max_estimate = self.domain[np.argmax(self.max_distribution)]
        return [min_estimate, max_estimate]

    def __copy__(self):
        copied_learner = type(self)(domain=copy(self.domain),
                                    value_distribution=copy(
                                        self.value_distribution),
                                    n_evidence_minmax=self.n_evidence_minmax)
        copied_learner.__dict__.update(self.__dict__)
        copied_learner.reset_learner()
        return copied_learner


class RandomMinMaxLearner(MinMaxLearner):
    def choose_query(self):
        timer = CodeTimer()
        with timer:
            if self.safe:
                fake_score = np.random.rand(self.value_distribution.shape[0])
                index = safe_selection(self.value_distribution, fake_score,
                                       self.tau_cutoff)
            else:
                index = np.random.randint(0, self.domain.shape[0] - 1)
            self.last_query = self.domain[index]

        if not self.profiling:
            return self.last_query
        else:
            query_lik = self.value_distribution[index]
            selection_results = {
                'query': self.last_query,
                'query_lik': query_lik,
                'time': timer.elapsed
            }
            return selection_results

    def update_model(self, answer, query=None):
        if not self.profiling:
            return super(RandomMinMaxLearner, self).update_model(answer, query)
        else:
            update_results = super(RandomMinMaxLearner,
                                   self).update_model(answer, query)
            return update_results


class WeightedBinarySearchMinMaxLearner(MinMaxLearner):
    def reset_learner(self):
        super(WeightedBinarySearchMinMaxLearner, self).reset_learner()
        self.target_distribution_max = True

    def choose_query(self):
        timer = CodeTimer()
        with timer:
            if self.safe:
                scores = compute_weighted_split_score(self.max_distribution) if self.target_distribution_max else \
                    compute_weighted_split_score(self.min_distribution)
                index = safe_selection(self.value_distribution,
                                       scores,
                                       self.tau_cutoff,
                                       score_function=np.argmin)
            else:
                index = compute_weighted_split(self.max_distribution) if self.target_distribution_max else \
                    compute_weighted_split(self.min_distribution)
            query = self.domain[index]
            self.target_distribution_max = not (self.target_distribution_max)
            self.last_query = query

        if not self.profiling:
            return self.last_query
        else:
            query_lik = self.value_distribution[index]
            selection_results = {
                'query': self.last_query,
                'query_lik': query_lik,
                'time': timer.elapsed
            }
            return selection_results


class BinarySearchMinMaxLearner(WeightedBinarySearchMinMaxLearner):
    def reset_learner(self):
        super(BinarySearchMinMaxLearner, self).reset_learner()
        # (re)compute related distributions
        try:
            n = self.domain.shape[0]
            self.max_distribution = np.ones(n) / n
            self.min_distribution = np.ones(n) / n
        except ValueError as error:
            raise error


class EntropyMinMaxLearner(MinMaxLearner):
    def choose_query(self):
        timer = CodeTimer()
        with timer:
            n = self.domain.shape[0]
            ere_max = np.zeros(n)
            ere_min = np.zeros(n)

            entropy_max_prior = entropy_categorical(self.max_distribution)
            entropy_min_prior = entropy_categorical(self.min_distribution)

            for j, value in enumerate(self.domain):
                ere_max[j] = compute_expected_reduction_in_entropy(
                    self.domain, self.max_distribution, value)
                ere_min[j] = compute_expected_reduction_in_entropy(
                    self.domain, self.min_distribution, value)

            if self.safe:
                index_max_prior = safe_selection(self.value_distribution,
                                                 ere_max - entropy_max_prior,
                                                 self.tau_cutoff,
                                                 score_function=np.argmin)
                index_min_prior = safe_selection(self.value_distribution,
                                                 ere_min - entropy_min_prior,
                                                 self.tau_cutoff,
                                                 score_function=np.argmin)

                index = index_max_prior if (ere_max - entropy_max_prior)[index_max_prior] <= \
                                           (ere_min - entropy_min_prior)[index_min_prior] else index_min_prior
            else:
                if np.min(ere_max -
                          entropy_max_prior) <= np.min(ere_min -
                                                       entropy_min_prior):
                    index = np.argmin(ere_max - entropy_max_prior)
                else:
                    index = np.argmin(ere_min - entropy_min_prior)
            query = self.domain[index]

            self.last_query = query

        if not self.profiling:
            return self.last_query
        else:
            query_lik = self.value_distribution[index]
            selection_results = {
                'query': self.last_query,
                'query_lik': query_lik,
                'ere_max': ere_max,
                'ere_min': ere_min,
                'entropy_max_prior': entropy_max_prior,
                'entropy_min_prior': entropy_min_prior,
                'time': timer.elapsed
            }
            return selection_results


class SafeEntropyMinMaxLearner(EntropyMinMaxLearner):
    def __init__(self, phi=1.0, *args, **kwargs):
        self.phi = phi
        super(SafeEntropyMinMaxLearner, self).__init__(*args, **kwargs)

    def choose_query(self):
        timer = CodeTimer()
        with timer:
            n = self.domain.shape[0]
            ere_max = np.zeros(n)
            ere_min = np.zeros(n)

            entropy_max_prior = entropy_categorical(self.max_distribution)
            entropy_min_prior = entropy_categorical(self.min_distribution)

            for j, value in enumerate(self.domain):
                ere_max[j] = compute_expected_reduction_in_entropy(
                    self.domain, self.max_distribution, value)
                ere_min[j] = compute_expected_reduction_in_entropy(
                    self.domain, self.min_distribution, value)

            ere_max = ere_max - entropy_max_prior
            ere_min = ere_min - entropy_min_prior

            # Squashes EREs and likelihood from 0 [worst] to 1 [best]
            common_max = np.max([np.max(ere_max), np.max(ere_min)])
            common_min = np.min([np.min(ere_max), np.min(ere_min)])
            ere_max = (ere_max - common_max) / (common_min - common_max)
            ere_min = (ere_min - common_max) / (common_min - common_max)
            likelihood = (self.value_distribution - np.min(self.value_distribution)) /\
                         (np.max(self.value_distribution) - np.min(self.value_distribution))

            # Create separated scores
            score_max = (likelihood**self.phi) * ere_max
            score_min = (likelihood**self.phi) * ere_min
            # score_max = self.phi * ere_max + (1 - self.phi) * likelihood
            # score_min = self.phi * ere_min + (1 - self.phi) * likelihood

            if np.max(score_max) >= np.max(score_min):
                index = np.argmax(score_max)
            else:
                index = np.argmax(score_min)
            query = self.domain[index]

            self.last_query = query

        if not self.profiling:
            return self.last_query
        else:
            query_lik = self.value_distribution[index]
            selection_results = {
                'query': self.last_query,
                'query_lik': query_lik,
                'score_max': score_max,
                'score_min': score_min,
                'time': timer.elapsed
            }
            return selection_results


class EntropySumMinMaxLearner(MinMaxLearner):
    def choose_query(self):
        timer = CodeTimer()
        with timer:
            n = self.domain.shape[0]
            ere_sum = np.zeros(n)
            entropy_sum_prior = entropy_categorical(
                self.min_distribution) + entropy_categorical(
                    self.max_distribution)

            for j, value in enumerate(self.domain):
                ere_sum[j] += compute_expected_reduction_in_entropy(
                    self.domain, self.max_distribution, value)
                ere_sum[j] += compute_expected_reduction_in_entropy(
                    self.domain, self.min_distribution, value)

            if self.safe:
                index = safe_selection(self.value_distribution,
                                       ere_sum - entropy_sum_prior,
                                       self.tau_cutoff,
                                       score_function=np.argmin)
            else:
                index = np.argmin(ere_sum - entropy_sum_prior)
            self.last_query = self.domain[index]

        if not self.profiling:
            return self.last_query
        else:
            query_lik = self.value_distribution[index]
            selection_results = {
                'query': self.last_query,
                'query_lik': query_lik,
                'ere_sum': ere_sum,
                'entropy_sum_prior': entropy_sum_prior,
                'time': timer.elapsed
            }
            return selection_results


class DivergenceMinMaxLearner(MinMaxLearner):
    def choose_query(self):
        timer = CodeTimer()
        with timer:
            n = self.domain.shape[0]
            erd_max = np.zeros(n)
            erd_min = np.zeros(n)

            for j, value in enumerate(self.domain):
                erd_max[j] = compute_expected_reduction_in_divergence(
                    self.domain, self.max_distribution, value)
                erd_min[j] = compute_expected_reduction_in_divergence(
                    self.domain, self.min_distribution, value)

            if self.safe:
                index_max_prior = safe_selection(self.value_distribution,
                                                 erd_max,
                                                 self.tau_cutoff,
                                                 score_function=np.argmax)
                index_min_prior = safe_selection(self.value_distribution,
                                                 erd_min,
                                                 self.tau_cutoff,
                                                 score_function=np.argmax)
                index = index_max_prior if erd_max[index_max_prior] >= erd_min[
                    index_min_prior] else index_min_prior
            else:
                if np.max(erd_max) >= np.max(erd_min):
                    index = np.argmax(erd_max)
                else:
                    index = np.argmax(erd_min)
            self.last_query = self.domain[index]

        if not self.profiling:
            return self.last_query
        else:
            query_lik = self.value_distribution[index]
            selection_results = {
                'query': self.last_query,
                'query_lik': query_lik,
                'erd_max': erd_max,
                'erd_min': erd_min,
                'time': timer.elapsed
            }
            return selection_results


class RangeLearner(Learner):
    def __init__(self,
                 n_evidence_range,
                 logistic_k_range=20.0,
                 *args,
                 **kwargs):
        self.logistic_k_range = logistic_k_range
        self.n_evidence_range = n_evidence_range
        super(RangeLearner, self).__init__(*args, **kwargs)

    @Learner.value_distribution.setter
    def value_distribution(self, value_distribution):
        super(RangeLearner,
              type(self)).value_distribution.fset(self, value_distribution)
        # (re)compute related distributions
        self.reset_learner()

    def reset_learner(self):
        # (re)compute related distributions
        try:
            super(RangeLearner, self).reset_learner()
            self.range_distribution = compute_sample_range_distribution(
                self.value_distribution, self.n_evidence_range)
            # add initial estimate (before any query) to the estimate history
            self.last_estimate = self.get_current_estimate()
        except ValueError as error:
            raise error

    def update_model(self, answer, query=None):
        timer = CodeTimer()

        with timer:
            q = self.last_query if query is None else query

            if q is None:
                raise LearnerException(2)

            if self.profiling:
                n = self.domain.shape[0]
                range_filter_test = np.ones([n, n]) / n**2

            if answer == LearnerAnswers.FINE:
                self.range_distribution = range_query_filter(
                    self.domain,
                    self.range_distribution,
                    q,
                    self.logistic_k_range,
                    is_range=True)
                if self.profiling:
                    range_filter_test = range_query_filter(
                        self.domain, range_filter_test, q,
                        self.logistic_k_range)
            elif answer == LearnerAnswers.HIGHER:
                self.range_distribution = range_query_filter(
                    self.domain,
                    self.range_distribution,
                    q,
                    self.logistic_k_range,
                    higher_max=True,
                    higher_min=True,
                    is_range=True)
                if self.profiling:
                    range_filter_test = range_query_filter(
                        self.domain,
                        range_filter_test,
                        q,
                        self.logistic_k_range,
                        higher_max=True,
                        higher_min=True)
            elif answer == LearnerAnswers.LOWER:
                self.range_distribution = range_query_filter(
                    self.domain,
                    self.range_distribution,
                    q,
                    self.logistic_k_range,
                    higher_max=False,
                    higher_min=False,
                    is_range=True)
                if self.profiling:
                    range_filter_test = range_query_filter(
                        self.domain,
                        range_filter_test,
                        q,
                        self.logistic_k_range,
                        higher_max=False,
                        higher_min=False)
            else:
                raise LearnerException(1)

            self.last_answer = answer
            self.last_query = None
            self.last_estimate = self.get_current_estimate()

        if not self.profiling:
            return True
        else:
            update_results = {
                'range_distribution': self.range_distribution,
                'range_filter_test': range_filter_test,
                'time': timer.elapsed
            }
            return update_results

    def get_current_estimate(self):
        indexes = np.unravel_index(
            np.argmax(self.range_distribution, axis=None),
            self.range_distribution.shape)
        return [self.domain[indexes[1]],
                self.domain[indexes[0]]]  # the max is on the x-axis...

    def __copy__(self):
        copied_learner = type(self)(domain=copy(self.domain),
                                    value_distribution=copy(
                                        self.value_distribution),
                                    n_evidence_range=self.n_evidence_range)
        copied_learner.__dict__.update(self.__dict__)
        copied_learner.reset_learner()
        return copied_learner


class EntropyRangeLearner(RangeLearner):
    def choose_query(self):
        timer = CodeTimer()
        with timer:
            n = self.domain.shape[0]
            ere_values = np.zeros(n)
            entropy_prior = entropy_categorical(self.range_distribution)

            for j, value in enumerate(self.domain):
                ere_values[j] = compute_expected_reduction_in_entropy(
                    self.domain, self.range_distribution, value)
            if self.safe:
                index = safe_selection(self.value_distribution,
                                       ere_values,
                                       self.tau_cutoff,
                                       score_function=np.argmin)
            else:
                index = np.argmin(ere_values)
            self.last_query = self.domain[index]

        if not self.profiling:
            return self.last_query
        else:
            query_lik = self.value_distribution[index]
            selection_results = {
                'query': self.last_query,
                'query_lik': query_lik,
                'ere_values': ere_values,
                'entropy_prior': entropy_prior,
                'time': timer.elapsed
            }
            return selection_results


class DivergenceRangeLearner(RangeLearner):
    def choose_query(self):
        timer = CodeTimer()
        with timer:
            n = self.domain.shape[0]
            divergences = np.zeros(n)

            for j, value in enumerate(self.domain):
                divergences[j] = compute_expected_reduction_in_divergence(
                    self.domain, self.range_distribution, value)

            if self.safe:
                index = safe_selection(self.value_distribution, divergences,
                                       self.tau_cutoff)
            else:
                index = np.argmax(divergences)
            self.last_query = self.domain[index]

        if not self.profiling:
            return self.last_query
        else:
            query_lik = self.value_distribution[index]
            selection_results = {
                'query': self.last_query,
                'query_lik': query_lik,
                'divergences': divergences,
                'time': timer.elapsed
            }
            return selection_results


class LearnerException(Exception):
    def __init__(self, error_id):
        error_messages = {
            0: 'value distribution not correctly initialized',
            1: 'unknown answer',
            2: 'could not find query'
        }
        message = error_messages.get(error_id, 'Unknown Error')
        super(LearnerException, self).__init__(message)


class LearnerAnswers(Enum):
    FINE = 0
    HIGHER = 1
    LOWER = 2


def simple_query_filter(domain,
                        function,
                        current_value,
                        logistic_k,
                        higher=True,
                        pdf=True):
    # The filter is a logistic function, centered around current_value with a certain steepness (logistic_k)
    # and direction (higher).
    # https://en.wikipedia.org/wiki/Logistic_function

    dom = np.linspace(0, 1, domain.shape[0])
    domain_size = domain[-1] - domain[0]
    dom -= (current_value - domain[0]) / domain_size
    dom *= logistic_k

    f = function * expit(dom) if higher else function * expit(-dom)
    f += np.finfo(function.dtype).eps

    if pdf:
        f /= np.sum(f)
        is_true_categorical_pdf(f)
        return f
    else:
        return f


def range_query_filter(domain,
                       function,
                       current_value,
                       logistic_k,
                       higher_max=True,
                       higher_min=False,
                       pdf=True,
                       is_range=False):
    # The filter is the 2-d version of simple_query_filter
    # https://math.stackexchange.com/questions/863662/need-function-for-2d-sigmoid-shaped-monotonic-surface

    range_filter = copy(function)
    max_filter = simple_query_filter(domain, np.ones(function.shape[0]),
                                     current_value, logistic_k, higher_max)
    min_filter = simple_query_filter(domain, np.ones(function.shape[1]),
                                     current_value, logistic_k, higher_min)

    for u in range(0, function.shape[0]):
        for v in range(0, function.shape[1]):
            range_filter[u, v] *= max_filter[u] * min_filter[v]

    if is_range:
        index = np.tril_indices(function.shape[0], -1)
        range_filter.T[index] = 0.0
    if pdf:
        range_filter /= np.sum(range_filter)
        assert np.isclose(np.sum(range_filter), 1.0)
    return range_filter


def generate_truth_range(domain, lowerbound, upperbound, amplitude=1.0):
    image = copy(domain)
    for i, value in enumerate(domain):
        if lowerbound <= value <= upperbound:
            image[i] = 1.0 * amplitude
        else:
            image[i] = 0.0
    return image


def is_true_categorical_pdf(categorical_pdf, strict=False):
    # Check if we are dealing with a categorical
    # if strict, raises sum error without trying to fix them
    # else it tries at least once
    try:
        assert np.isclose(np.sum(categorical_pdf), 1.0)
    except AssertionError:
        if strict:
            print('pdf integral not summing to 1 but to {}...'.format(
                np.sum(categorical_pdf)))
            raise
        else:
            categorical_pdf /= np.sum(categorical_pdf)
            is_true_categorical_pdf(categorical_pdf, strict=True)
    try:
        # put values under machine precision (between -eps and 0.0) to 0.0
        # they are zeros, so I don't want them to trigger the assertion
        categorical_pdf[np.where(
            np.isclose(categorical_pdf[np.where(categorical_pdf < 0.0)],
                       0.0))] = 0.0
        assert np.alltrue(categorical_pdf[np.where(
            np.invert(np.isclose(categorical_pdf, 0.0)))] >= 0)
    except AssertionError:
        print('pdf contains negative probabilities...')
        print('apparently pdf contains has {}'.format(
            categorical_pdf[np.where(categorical_pdf < 0.0)]))
        raise
    return True


def entropy_categorical(categorical_pdf):
    if len(categorical_pdf.shape) > 1:
        categorical_pdf = categorical_pdf.flatten()

    is_true_categorical_pdf(categorical_pdf)
    entropy = np.sum([
        -value * np.log2(value) for value in categorical_pdf
        if not (value < np.finfo(categorical_pdf.dtype).eps)
    ])
    try:
        assert entropy > 0
    except AssertionError:
        print('weird - entropy is positive...')
        raise
    return entropy


def kl_divergence_categorical(first_pdf, second_pdf):
    try:
        assert first_pdf.shape == second_pdf.shape
    except AssertionError:
        print('the two pdfs have different dimensionality!')
        raise

    if len(first_pdf.shape) > 1:
        first_pdf = first_pdf.flatten()
    if len(second_pdf.shape) > 1:
        second_pdf = second_pdf.flatten()

    # Check for ill-defined pdfs
    is_true_categorical_pdf(first_pdf)
    is_true_categorical_pdf(second_pdf)

    with np.errstate(divide='ignore', invalid='ignore'):
        return np.sum(
            np.where(first_pdf < np.finfo(first_pdf.dtype).eps, 0.0,
                     first_pdf * np.log2(first_pdf / second_pdf)))


def js_divergence_categorical(first_pdf, second_pdf):
    try:
        assert first_pdf.shape == second_pdf.shape
    except AssertionError:
        print('the two pdfs have different dimensionality!')
        raise

    # Check for ill-defined pdfs
    is_true_categorical_pdf(first_pdf)
    is_true_categorical_pdf(second_pdf)

    sum_pdf = 0.5 * (first_pdf + second_pdf)
    is_true_categorical_pdf(sum_pdf)

    return 0.5 * (kl_divergence_categorical(first_pdf, sum_pdf) +
                  kl_divergence_categorical(second_pdf, sum_pdf))


def numerical_integration(domain, function, lower_limit, upper_limit):
    if len(function.shape) == 1:
        index_lower = None
        index_upper = None
        for i, value in enumerate(domain):
            if value == lower_limit:
                index_lower = i
            if value == upper_limit:
                index_upper = i
        if index_lower is None or index_upper is None:
            print('Cannot find lower or upper bounds')
            raise ValueError
        return np.sum(function[index_lower:index_upper + 1])
    elif len(function.shape) == 2:
        if len(lower_limit) != 2 or len(upper_limit) != 2:
            print('Integration limits should be specified for each dimension')
            raise ValueError
        i_x_lower = None
        i_x_upper = None
        i_y_lower = None
        i_y_upper = None
        for i, value in enumerate(domain):
            if value == lower_limit[0]:
                i_x_lower = i
            if value == upper_limit[0]:
                i_x_upper = i
            if value == lower_limit[1]:
                i_y_lower = i
            if value == upper_limit[1]:
                i_y_upper = i
        if i_x_lower is None or i_x_upper is None or i_y_lower is None or i_y_upper is None:
            print('Cannot find lower or upper bounds')
            raise ValueError
        return np.sum(function[i_x_lower:i_x_upper + 1,
                               i_y_lower:i_y_upper + 1])
    else:
        print('Can only integrate up to 2 dimensions')
        raise ValueError


def compute_expected_reduction_in_entropy(domain, function, query_value):
    if len(function.shape) == 1:
        posterior_higher = simple_query_filter(domain, function, query_value,
                                               50.0, True)
        posterior_lower = simple_query_filter(domain, function, query_value,
                                              50.0, False)

        probs = np.zeros(2)
        probs[0] = numerical_integration(domain, function, query_value,
                                         domain[-1])  # Higher
        probs[1] = numerical_integration(domain, function, domain[0],
                                         query_value)  # Lower
        # TODO: actually, no need to do the second integration - just do probs[1] = 1 -probs[0] IF the function is normalized to begin with
        probs /= np.sum(probs)

        is_true_categorical_pdf(probs)

        entropies = np.zeros(2)
        entropies[0] = entropy_categorical(posterior_higher)
        entropies[1] = entropy_categorical(posterior_lower)

        return np.dot(probs, entropies)
    elif len(function.shape) == 2:
        posterior_fine = range_query_filter(domain, function, query_value,
                                            20.0)
        posterior_higher = range_query_filter(domain,
                                              function,
                                              query_value,
                                              20.0,
                                              higher_max=True,
                                              higher_min=True)
        posterior_lower = range_query_filter(domain,
                                             function,
                                             query_value,
                                             20.0,
                                             higher_max=False,
                                             higher_min=False)

        probs = np.zeros(3)
        # in the order, Fine, Higher, Lower
        probs[0] = numerical_integration(domain, function,
                                         [query_value, domain[0]],
                                         [domain[-1], query_value])
        probs[1] = numerical_integration(domain, function,
                                         [query_value, query_value],
                                         [domain[-1], domain[-1]])
        probs[2] = numerical_integration(domain, function,
                                         [domain[0], domain[0]],
                                         [query_value, query_value])
        probs /= np.sum(probs)

        is_true_categorical_pdf(probs)

        entropies = np.zeros(3)
        entropies[0] = entropy_categorical(posterior_fine)
        entropies[1] = entropy_categorical(posterior_higher)
        entropies[2] = entropy_categorical(posterior_lower)

        return np.dot(probs, entropies)
    else:
        print('Can only do this for up to 2 dimensions, not for {}'.format(
            len(function.shape)))
        raise ValueError


def compute_expected_reduction_in_divergence(domain,
                                             function,
                                             query_value,
                                             method='js'):
    if method == 'js':
        chosen_method = js_divergence_categorical
    elif method == 'kl':
        chosen_method = kl_divergence_categorical
    else:
        print(
            'Unknown divergence method. Known methods: Jensen-Shannon js, Kullback-Leibler kl.\n'
            + 'Using Jensen-Shannon...')
        chosen_method = js_divergence_categorical

    if len(function.shape) == 1:
        posterior_higher = simple_query_filter(domain, function, query_value,
                                               50.0, True)
        posterior_lower = simple_query_filter(domain, function, query_value,
                                              50.0, False)

        is_true_categorical_pdf(posterior_higher)
        is_true_categorical_pdf(posterior_lower)

        probs = np.zeros(2)
        probs[0] = numerical_integration(domain, function, query_value,
                                         domain[-1])  # Higher
        probs[1] = numerical_integration(domain, function, domain[0],
                                         query_value)  # Lower
        probs /= np.sum(probs)

        is_true_categorical_pdf(probs)

        divergences = np.zeros(2)
        divergences[0] = chosen_method(posterior_higher, function)
        divergences[1] = chosen_method(posterior_lower, function)

        return np.dot(probs, divergences)
    elif len(function.shape) == 2:
        posterior_fine = range_query_filter(domain, function, query_value,
                                            20.0)
        posterior_higher = range_query_filter(domain,
                                              function,
                                              query_value,
                                              20.0,
                                              higher_max=True,
                                              higher_min=True)
        posterior_lower = range_query_filter(domain,
                                             function,
                                             query_value,
                                             20.0,
                                             higher_max=False,
                                             higher_min=False)

        probs = np.zeros(3)
        # in the order, Fine, Higher and Lower
        probs[0] = numerical_integration(domain, function,
                                         [query_value, domain[0]],
                                         [domain[-1], query_value])
        probs[1] = numerical_integration(domain, function,
                                         [query_value, query_value],
                                         [domain[-1], domain[-1]])
        probs[2] = numerical_integration(domain, function,
                                         [domain[0], domain[0]],
                                         [query_value, query_value])
        probs /= np.sum(probs)

        is_true_categorical_pdf(probs)

        divergences = np.zeros(3)
        divergences[0] = chosen_method(posterior_fine, function)
        divergences[1] = chosen_method(posterior_higher, function)
        divergences[2] = chosen_method(posterior_lower, function)

        return np.dot(probs, divergences)
    else:
        print('Can only do this for up to 2 dimensions, not for {}'.format(
            len(function.shape)))
        raise ValueError


def compute_sample_max_distribution(pdf, n_evidence):
    if n_evidence < 1:
        raise ValueError('n_evidence must be greater than 0')

    cdf = np.cumsum(pdf)
    sample_max_distribution = n_evidence * cdf**(n_evidence - 1) * pdf
    return sample_max_distribution / np.sum(sample_max_distribution)


def compute_sample_min_distribution(pdf, n_evidence):
    if n_evidence < 1:
        raise ValueError('n_evidence must be greater than 0')

    cdf = np.cumsum(pdf)
    sample_min_distribution = n_evidence * (1 - cdf)**(n_evidence - 1) * pdf
    return sample_min_distribution / np.sum(sample_min_distribution)


def compute_sample_range_distribution(pdf, n_evidence):
    if n_evidence < 2:
        raise ValueError('n_evidence must be greater than 1')

    cdf = np.cumsum(pdf)
    range_pdf = np.zeros((pdf.shape[0], pdf.shape[0]))

    for u in range(0, pdf.shape[0]):
        for v in range(0, pdf.shape[0]):
            range_pdf[u, v] = n_evidence * (n_evidence - 1) * (
                cdf[u] - cdf[v])**(n_evidence - 2) * pdf[u] * pdf[v]
    index = np.tril_indices(pdf.shape[0], -1)
    range_pdf.T[index] = 0.0
    return range_pdf / np.sum(range_pdf)


def compute_weighted_split(function, sigma=0.5):
    # returns index in vector function where the integral from the beginning to index is sigma*np.sum(function)
    integral = np.sum(function)
    current_sum = 0.0
    i = -1
    for i, value in enumerate(function):
        current_sum += value
        if current_sum >= sigma * integral:
            break
    return i


def compute_weighted_split_score(function):
    # returns scores for the split method, see paper
    prob_mass = np.sum(function)
    cdf = np.cumsum(function)
    inverse_cdf = prob_mass * np.ones(function.shape) - cdf
    return np.abs(cdf - inverse_cdf)


def compute_distribution_tail_cutoff(distribution, phi):
    # return the value tau so that the lowest phi fraction (i.e. phi*100 %) of the distribution is cut off
    # in other words, it finds the value so that if p(x)<=tau, then x is in the less likely part of the dist (in tails)
    if not (0.0 <= phi <= 1.0):
        raise ValueError('phi must be [0,1], it is {}'.format(phi))
    if np.isclose(phi, 0.0):
        return 0.0
    if np.isclose(phi, 1.0):
        return np.max(distribution)
    sorted_distribution = np.sort(distribution)
    cumsum_distribution = np.cumsum(sorted_distribution)
    return sorted_distribution[np.argmax(cumsum_distribution >= phi)]


def safe_selection(distribution,
                   score,
                   distribution_cutoff,
                   cutoff_function=np.greater_equal,
                   score_function=np.argmax):
    # returns the index of score_function(score)
    # that ALSO satisfies distribution(x) 'cutoff_function' distribution_cutoff
    mask = cutoff_function(distribution, distribution_cutoff)
    index = np.arange(0, score.shape[0])
    temp = (index[mask])[score_function(score[mask])]
    return temp


def generate_prior(seed, domain, beta_weights=[2, 10]):

    np.random.seed(seed)
    bin_size = domain[1] - domain[0]

    # Create prior distribution of parameter value
    first_prior = beta(np.random.randint(beta_weights[0], beta_weights[1]),
                       np.random.randint(beta_weights[0], beta_weights[1]))
    second_prior = beta(np.random.randint(beta_weights[0], beta_weights[1]),
                        np.random.randint(beta_weights[0], beta_weights[1]))
    third_prior = beta(np.random.randint(beta_weights[0], beta_weights[1]),
                       np.random.randint(beta_weights[0], beta_weights[1]))
    forth_prior = beta(np.random.randint(beta_weights[0], beta_weights[1]),
                       np.random.randint(beta_weights[0], beta_weights[1]))
    w = np.random.dirichlet((1, 1, 1, 1))

    prior = []
    for value in domain:
        prior.append(w[0] * first_prior.pdf(value) +
                     w[1] * second_prior.pdf(value) +
                     w[2] * third_prior.pdf(value) +
                     w[3] * forth_prior.pdf(value))

    prior = np.array(prior)
    prior /= np.sum(prior)

    # Create ground truth range
    upperbound_prior = compute_sample_max_distribution(prior, 2)
    lowerbound_prior = compute_sample_min_distribution(prior, 2)
    lowerbound = domain[np.nonzero(np.random.multinomial(1, lowerbound_prior))]
    filtered_upperbound_prior = simple_query_filter(domain, upperbound_prior,
                                                    lowerbound + bin_size,
                                                    1000.0, True)
    upperbound = domain[np.nonzero(
        np.random.multinomial(1, filtered_upperbound_prior))]

    return prior, lowerbound, upperbound, lowerbound_prior, upperbound_prior


class CodeTimer:
    def __init__(self, name=None):
        self.name = " '" + name + "'" if name else ''

    def __enter__(self):
        self.start = time()

    def __exit__(self, exc_type, exc_value, traceback):
        self.elapsed = (time() - self.start)
