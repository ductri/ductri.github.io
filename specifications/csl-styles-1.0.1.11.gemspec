# -*- encoding: utf-8 -*-
# stub: csl-styles 1.0.1.11 ruby lib

Gem::Specification.new do |s|
  s.name = "csl-styles".freeze
  s.version = "1.0.1.11".freeze

  s.required_rubygems_version = Gem::Requirement.new(">= 0".freeze) if s.respond_to? :required_rubygems_version=
  s.require_paths = ["lib".freeze]
  s.authors = ["Sylvester Keil".freeze]
  s.date = "2021-05-10"
  s.description = "\n    The official Citation Style Language (CSL) styles and locale files.\n    ".freeze
  s.email = ["http://sylvester.keil.or.at".freeze]
  s.homepage = "https://github.com/inukshuk/csl-styles".freeze
  s.licenses = ["CC-BY-SA-3.0".freeze]
  s.rubygems_version = "3.0.3".freeze
  s.summary = "CSL styles and locales".freeze

  s.installed_by_version = "3.5.16".freeze if s.respond_to? :installed_by_version

  s.specification_version = 4

  s.add_runtime_dependency(%q<csl>.freeze, ["~> 1.0".freeze])
end
