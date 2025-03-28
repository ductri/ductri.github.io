# -*- encoding: utf-8 -*-
# stub: moving 0.1.0 ruby lib

Gem::Specification.new do |s|
  s.name = "moving".freeze
  s.version = "0.1.0".freeze

  s.required_rubygems_version = Gem::Requirement.new(">= 0".freeze) if s.respond_to? :required_rubygems_version=
  s.require_paths = ["lib".freeze]
  s.authors = ["huangyz0918".freeze]
  s.date = "2019-08-25"
  s.email = ["huangyz0918@gmail.com".freeze]
  s.homepage = "https://github.com/huangyz0918/moving".freeze
  s.licenses = ["MIT".freeze]
  s.rubygems_version = "3.0.6".freeze
  s.summary = "A clean and minimalist theme for Jekyll".freeze

  s.installed_by_version = "3.5.16".freeze if s.respond_to? :installed_by_version

  s.specification_version = 4

  s.add_runtime_dependency(%q<jekyll>.freeze, ["~> 4.0".freeze])
  s.add_runtime_dependency(%q<jekyll-feed>.freeze, ["~> 0.9".freeze])
  s.add_runtime_dependency(%q<jekyll-seo-tag>.freeze, ["~> 2.1".freeze])
  s.add_development_dependency(%q<bundler>.freeze, ["~> 1.16".freeze])
  s.add_development_dependency(%q<rake>.freeze, ["~> 12.0".freeze])
end
