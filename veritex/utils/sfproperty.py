import sys
from veritex.sets.boxdomain import BoxDomain

class Property:
    def __init__(self, input_domain: list, unsafe_output_domains: list, input_ranges=None, set_type='facet-vertex'):
        assert len(input_domain)!=0
        self.lbs = input_domain[0]
        self.ubs = input_domain[1]

        self.set_type = set_type
        self.construct_input()
        self.unsafe_domains = unsafe_output_domains
        self.input_ranges = input_ranges


    def construct_input(self):
        box = BoxDomain(self.lbs, self.ubs)
        if self.set_type=='facet-vertex':
            self.input_set = box.toFacetVertex()
        else:
            sys.exit("This set type is not supported.")
