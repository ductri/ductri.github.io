#
# DO NOT MODIFY!!!!
# This file is automatically generated by Racc 1.7.3
# from Racc grammar file "parser.y".
#

require 'racc/parser.rb'

require 'strscan'

module Namae
  class Parser < Racc::Parser

module_eval(<<'...end parser.y/module_eval...', 'parser.y', 111)

  @defaults = {
    :debug => false,
    :prefer_comma_as_separator => false,
    :include_particle_in_family => false,
    :comma => ',',
    :stops => ',;',
    :separator => /\s*(\band\b|\&|;)\s*/i,
    :title => /\s*\b(sir|lord|count(ess)?|(gen|adm|col|maj|capt|cmdr|lt|sgt|cpl|pvt|pastor|pr|reverend|rev|elder|deacon|deaconess|father|fr|rabbi|cantor|vicar|prof|dr|md|ph\.?d)\.?)(\s+|$)/i,
    :suffix => /\s*\b(JR|Jr|jr|SR|Sr|sr|[IVX]{2,})(\.|\b)/,
    :appellation => /\s*\b((mrs?|ms|fr|hr)\.?|miss|herr|frau)(\s+|$)/i,
    :uppercase_particle => /\s*\b(D[aiu]|De[rs]?|St\.?|Saint|La|Les|V[ao]n)(\s+|$)/
  }

  class << self
    attr_reader :defaults

    def instance
      Thread.current[:namae] ||= new
    end
  end

  attr_reader :options, :input

  def initialize(options = {})
    @options = self.class.defaults.merge(options)
  end

  def debug?
    options[:debug] || ENV['DEBUG']
  end

  def separator
    options[:separator]
  end

  def comma
    options[:comma]
  end

  def include_particle_in_family?
    options[:include_particle_in_family]
  end

  def stops
    options[:stops]
  end

  def title
    options[:title]
  end

  def suffix
    options[:suffix]
  end

  def appellation
    options[:appellation]
  end

  def uppercase_particle
    options[:uppercase_particle]
  end

  def prefer_comma_as_separator?
    options[:prefer_comma_as_separator]
  end

  def parse(string)
    parse!(string)
  rescue => e
    warn e.message if debug?
    []
  end

  def parse!(string)
    @input = StringScanner.new(normalize(string))
    reset
    names = do_parse
    names.map(&:merge_particles!) if include_particle_in_family?
    names
  end

  def normalize(string)
    string.scrub.strip
  end

  def reset
    @commas, @words, @initials, @suffices, @yydebug = 0, 0, 0, 0, debug?
    self
  end

  private

  def stack
    @vstack || @racc_vstack || []
  end

  def last_token
    stack[-1]
  end

  def consume_separator
    return next_token if seen_separator?
    @commas, @words, @initials, @suffices = 0, 0, 0, 0
    [:AND, :AND]
  end

  def consume_comma
    @commas += 1
    [:COMMA, :COMMA]
  end

  def consume_word(type, word)
    @words += 1

    case type
    when :UWORD
      @initials += 1 if word =~ /^[[:upper:]]+\b/
    when :SUFFIX
      @suffices += 1
    end

    [type, word]
  end

  def seen_separator?
    !stack.empty? && last_token == :AND
  end

  def suffix?
    !@suffices.zero? || will_see_suffix?
  end

  def will_see_suffix?
    input.rest.strip.split(/\s+/)[0] =~ suffix
  end

  def will_see_initial?
    input.rest.strip.split(/\s+/)[0] =~ /^[[:upper:]]+\b/
  end

  def seen_full_name?
    prefer_comma_as_separator? && @words > 1 &&
      (@initials > 0 || !will_see_initial?) && !will_see_suffix?
  end

  def next_token
    case
    when input.nil?, input.eos?
      nil
    when input.scan(separator)
      consume_separator
    when input.scan(/\s*#{comma}\s*/)
      if @commas.zero? && !seen_full_name? || @commas == 1 && suffix?
        consume_comma
      else
        consume_separator
      end
    when input.scan(/\s+/)
      next_token
    when input.scan(title)
      consume_word(:TITLE, input.matched.strip)
    when input.scan(suffix)
      consume_word(:SUFFIX, input.matched.strip)
    when input.scan(appellation)
      if @words.zero?
        [:APPELLATION, input.matched.strip]
      else
        consume_word(:UWORD, input.matched)
      end
    when input.scan(uppercase_particle)
      consume_word(:UPARTICLE, input.matched.strip)
    when input.scan(/((\\\w+)?\{[^\}]*\})*[[:upper:]][^\s#{stops}]*/)
      consume_word(:UWORD, input.matched)
    when input.scan(/((\\\w+)?\{[^\}]*\})*[[:lower:]][^\s#{stops}]*/)
      consume_word(:LWORD, input.matched)
    when input.scan(/(\\\w+)?\{[^\}]*\}[^\s#{stops}]*/)
      consume_word(:PWORD, input.matched)
    when input.scan(/('[^'\n]+')|("[^"\n]+")/)
      consume_word(:NICK, input.matched[1...-1])
    else
      raise ArgumentError,
        "Failed to parse name #{input.string.inspect}: unmatched data at offset #{input.pos}"
    end
  end

  def on_error(tid, value, stack)
    raise ArgumentError,
      "Failed to parse name: unexpected '#{value}' at #{stack.inspect}"
  end

# -*- racc -*-
...end parser.y/module_eval...
##### State transition tables begin ###

racc_action_table = [
   -43,    36,    26,    37,   -41,    38,    39,   -43,   -42,   -43,
   -43,   -41,   -40,   -41,   -41,   -42,    45,   -42,   -42,   -40,
    50,   -40,   -40,    72,    59,    58,    60,    73,    16,    13,
    17,   -36,    61,     7,    18,    65,    14,    16,    25,    17,
    16,    25,    17,    28,    18,    14,    65,    45,    14,    36,
    34,    37,    68,    16,    13,    17,    26,    35,     7,    18,
    18,    14,    16,    25,    17,    28,    36,    34,    37,    45,
    14,    36,    34,    37,    35,    36,    34,    37,    45,    35,
    36,    52,    37,    35,   -22,   -22,   -22,    18,    35,    59,
    58,    60,   -22,    36,    34,    37,    45,    61,    36,    34,
    37,    35,    59,    58,    60,    65,    35,   nil,   nil,    45,
    61,    59,    58,    60,    59,    58,    60,   nil,    45,    61,
    19,   nil,    61,    59,    58,    60,   -40,    20,   -24,   nil,
   nil,    61,   nil,   -40 ]

racc_action_check = [
    14,    48,     8,    48,    16,    11,    19,    14,    17,    14,
    14,    16,    25,    16,    16,    17,    27,    17,    17,    25,
    31,    25,    25,    55,    55,    55,    55,    56,     0,     0,
     0,    55,    55,     0,     0,    56,     0,     5,     5,     5,
     9,     9,     9,     9,    43,     5,    44,    46,     9,    10,
    10,    10,    49,    20,    20,    20,    64,    10,    20,    20,
    66,    20,    23,    23,    23,    23,    24,    24,    24,    67,
    23,    28,    28,    28,    24,    29,    29,    29,    70,    28,
    33,    33,    33,    29,    34,    34,    34,    75,    33,    38,
    38,    38,    34,    41,    41,    41,    38,    38,    47,    47,
    47,    41,    50,    50,    50,    77,    47,   nil,   nil,    50,
    50,    68,    68,    68,    73,    73,    73,   nil,    68,    68,
     1,   nil,    73,    78,    78,    78,    13,     1,    13,   nil,
   nil,    78,   nil,    13 ]

racc_action_pointer = [
    25,   120,   nil,   nil,   nil,    34,   nil,   nil,    -7,    37,
    46,     3,   nil,   126,     0,   nil,     4,     8,   nil,     6,
    50,   nil,   nil,    59,    63,    12,   nil,     6,    68,    72,
   nil,    18,   nil,    77,    81,   nil,   nil,   nil,    86,   nil,
   nil,    90,   nil,    35,    36,   nil,    37,    95,    -2,    50,
    99,   nil,   nil,   nil,   nil,    21,    25,   nil,   nil,   nil,
   nil,   nil,   nil,   nil,    47,   nil,    51,    59,   108,   nil,
    68,   nil,   nil,   111,   nil,    78,   nil,    95,   120,   nil ]

racc_action_default = [
    -1,   -52,    -2,    -4,    -5,   -52,    -8,    -9,   -10,   -25,
   -52,   -52,   -19,   -22,   -23,   -30,   -32,   -33,   -50,   -52,
   -52,    -6,    -7,   -52,   -52,   -22,   -51,   -44,   -52,   -52,
   -31,   -15,   -20,   -25,   -24,   -23,   -32,   -33,   -38,    80,
    -3,   -52,   -15,   -48,   -45,   -46,   -44,   -52,   -25,   -14,
   -38,   -21,   -22,   -16,   -26,   -39,   -28,   -34,   -40,   -41,
   -42,   -43,   -14,   -11,   -49,   -47,   -48,   -44,   -38,   -17,
   -52,   -35,   -37,   -52,   -12,   -48,   -18,   -27,   -29,   -13 ]

racc_goto_table = [
     3,    30,    43,     1,    22,    21,    56,    53,    31,    27,
    32,    63,    78,    70,   nil,    30,   nil,   nil,    56,    69,
     3,    66,    42,    27,    32,    30,    46,    49,    24,    32,
     9,   nil,    29,    51,    74,    23,    56,    76,    77,    62,
    30,    32,    75,    79,     2,    67,    41,    32,     8,   nil,
     9,    47,   nil,   nil,   nil,    71,   nil,   nil,    48,   nil,
   nil,   nil,   nil,   nil,    40,   nil,   nil,   nil,     8,   nil,
   nil,   nil,   nil,   nil,   nil,   nil,   nil,   nil,    71 ]

racc_goto_check = [
     3,    19,     9,     1,     4,     3,    18,    13,    11,     3,
    14,    10,    16,    17,   nil,    19,   nil,   nil,    18,    13,
     3,     9,    11,     3,    14,    19,    11,    11,    12,    14,
     8,   nil,    12,    14,    10,     8,    18,    13,    18,    11,
    19,    14,     9,    10,     2,    11,    12,    14,     7,   nil,
     8,    12,   nil,   nil,   nil,     3,   nil,   nil,     8,   nil,
   nil,   nil,   nil,   nil,     2,   nil,   nil,   nil,     7,   nil,
   nil,   nil,   nil,   nil,   nil,   nil,   nil,   nil,     3 ]

racc_goto_pointer = [
   nil,     3,    44,     0,    -1,   nil,   nil,    48,    30,   -25,
   -32,    -2,    23,   -31,     0,   nil,   -61,   -42,   -32,    -8 ]

racc_goto_default = [
   nil,   nil,   nil,    57,     4,     5,     6,    64,    33,   nil,
   nil,    11,    10,   nil,    12,    54,    55,   nil,    44,    15 ]

racc_reduce_table = [
  0, 0, :racc_error,
  0, 13, :_reduce_1,
  1, 13, :_reduce_2,
  3, 13, :_reduce_3,
  1, 14, :_reduce_4,
  1, 14, :_reduce_none,
  2, 14, :_reduce_6,
  2, 14, :_reduce_7,
  1, 14, :_reduce_none,
  1, 17, :_reduce_9,
  1, 17, :_reduce_10,
  4, 16, :_reduce_11,
  5, 16, :_reduce_12,
  6, 16, :_reduce_13,
  3, 16, :_reduce_14,
  2, 16, :_reduce_15,
  3, 18, :_reduce_16,
  4, 18, :_reduce_17,
  5, 18, :_reduce_18,
  1, 24, :_reduce_none,
  2, 24, :_reduce_20,
  3, 24, :_reduce_21,
  1, 26, :_reduce_none,
  1, 26, :_reduce_none,
  1, 23, :_reduce_none,
  1, 23, :_reduce_none,
  1, 25, :_reduce_26,
  3, 25, :_reduce_27,
  1, 25, :_reduce_28,
  3, 25, :_reduce_29,
  1, 20, :_reduce_none,
  2, 20, :_reduce_31,
  1, 31, :_reduce_none,
  1, 31, :_reduce_none,
  1, 28, :_reduce_none,
  2, 28, :_reduce_35,
  0, 29, :_reduce_none,
  1, 29, :_reduce_none,
  0, 27, :_reduce_none,
  1, 27, :_reduce_none,
  1, 15, :_reduce_none,
  1, 15, :_reduce_none,
  1, 15, :_reduce_none,
  1, 15, :_reduce_none,
  0, 21, :_reduce_none,
  1, 21, :_reduce_none,
  1, 30, :_reduce_none,
  2, 30, :_reduce_47,
  0, 22, :_reduce_none,
  1, 22, :_reduce_none,
  1, 19, :_reduce_none,
  2, 19, :_reduce_51 ]

racc_reduce_n = 52

racc_shift_n = 80

racc_token_table = {
  false => 0,
  :error => 1,
  :COMMA => 2,
  :UWORD => 3,
  :LWORD => 4,
  :PWORD => 5,
  :NICK => 6,
  :AND => 7,
  :APPELLATION => 8,
  :TITLE => 9,
  :SUFFIX => 10,
  :UPARTICLE => 11 }

racc_nt_base = 12

racc_use_result_var = true

Racc_arg = [
  racc_action_table,
  racc_action_check,
  racc_action_default,
  racc_action_pointer,
  racc_goto_table,
  racc_goto_check,
  racc_goto_default,
  racc_goto_pointer,
  racc_nt_base,
  racc_reduce_table,
  racc_token_table,
  racc_shift_n,
  racc_reduce_n,
  racc_use_result_var ]
Ractor.make_shareable(Racc_arg) if defined?(Ractor)

Racc_token_to_s_table = [
  "$end",
  "error",
  "COMMA",
  "UWORD",
  "LWORD",
  "PWORD",
  "NICK",
  "AND",
  "APPELLATION",
  "TITLE",
  "SUFFIX",
  "UPARTICLE",
  "$start",
  "names",
  "name",
  "word",
  "display_order",
  "honorific",
  "sort_order",
  "titles",
  "u_words",
  "opt_suffices",
  "opt_titles",
  "last",
  "von",
  "first",
  "particle",
  "opt_words",
  "words",
  "opt_comma",
  "suffices",
  "u_word" ]
Ractor.make_shareable(Racc_token_to_s_table) if defined?(Ractor)

Racc_debug_parser = false

##### State transition tables end #####

# reduce 0 omitted

module_eval(<<'.,.,', 'parser.y', 11)
  def _reduce_1(val, _values, result)
     result = []
    result
  end
.,.,

module_eval(<<'.,.,', 'parser.y', 12)
  def _reduce_2(val, _values, result)
     result = [val[0]]
    result
  end
.,.,

module_eval(<<'.,.,', 'parser.y', 13)
  def _reduce_3(val, _values, result)
     result = val[0] << val[2]
    result
  end
.,.,

module_eval(<<'.,.,', 'parser.y', 15)
  def _reduce_4(val, _values, result)
     result = Name.new(:given => val[0])
    result
  end
.,.,

# reduce 5 omitted

module_eval(<<'.,.,', 'parser.y', 17)
  def _reduce_6(val, _values, result)
     result = val[0].merge(:family => val[1])
    result
  end
.,.,

module_eval(<<'.,.,', 'parser.y', 18)
  def _reduce_7(val, _values, result)
     result = val[1].merge(val[0])
    result
  end
.,.,

# reduce 8 omitted

module_eval(<<'.,.,', 'parser.y', 21)
  def _reduce_9(val, _values, result)
     result = Name.new(:appellation => val[0])
    result
  end
.,.,

module_eval(<<'.,.,', 'parser.y', 22)
  def _reduce_10(val, _values, result)
     result = Name.new(:title => val[0])
    result
  end
.,.,

module_eval(<<'.,.,', 'parser.y', 26)
  def _reduce_11(val, _values, result)
             result = Name.new(
           :given => val[0], :family => val[1], :suffix => val[2], :title => val[3]
         )

    result
  end
.,.,

module_eval(<<'.,.,', 'parser.y', 32)
  def _reduce_12(val, _values, result)
             result = Name.new(
           :given => val[0], :nick => val[1], :family => val[2], :suffix => val[3], :title => val[4]
         )

    result
  end
.,.,

module_eval(<<'.,.,', 'parser.y', 38)
  def _reduce_13(val, _values, result)
             result = Name.new(
           :given => val[0], :nick => val[1], :particle => val[2], :family => val[3], :suffix => val[4], :title => val[5])

    result
  end
.,.,

module_eval(<<'.,.,', 'parser.y', 43)
  def _reduce_14(val, _values, result)
             result = Name.new(:given => val[0], :particle => val[1], :family => val[2])

    result
  end
.,.,

module_eval(<<'.,.,', 'parser.y', 47)
  def _reduce_15(val, _values, result)
             result = Name.new(:particle => val[0], :family => val[1])

    result
  end
.,.,

module_eval(<<'.,.,', 'parser.y', 52)
  def _reduce_16(val, _values, result)
             result = Name.new({
           :family => val[0], :suffix => val[2][0], :given => val[2][1]
         }, !!val[2][0])

    result
  end
.,.,

module_eval(<<'.,.,', 'parser.y', 58)
  def _reduce_17(val, _values, result)
             result = Name.new({
           :particle => val[0], :family => val[1], :suffix => val[3][0], :given => val[3][1]
         }, !!val[3][0])

    result
  end
.,.,

module_eval(<<'.,.,', 'parser.y', 64)
  def _reduce_18(val, _values, result)
             result = Name.new({
           :particle => val[0,2].join(' '), :family => val[2], :suffix => val[4][0], :given => val[4][1]
         }, !!val[4][0])

    result
  end
.,.,

# reduce 19 omitted

module_eval(<<'.,.,', 'parser.y', 71)
  def _reduce_20(val, _values, result)
     result = val.join(' ')
    result
  end
.,.,

module_eval(<<'.,.,', 'parser.y', 72)
  def _reduce_21(val, _values, result)
     result = val.join(' ')
    result
  end
.,.,

# reduce 22 omitted

# reduce 23 omitted

# reduce 24 omitted

# reduce 25 omitted

module_eval(<<'.,.,', 'parser.y', 78)
  def _reduce_26(val, _values, result)
     result = [nil,val[0]]
    result
  end
.,.,

module_eval(<<'.,.,', 'parser.y', 79)
  def _reduce_27(val, _values, result)
     result = [val[2],val[0]]
    result
  end
.,.,

module_eval(<<'.,.,', 'parser.y', 80)
  def _reduce_28(val, _values, result)
     result = [val[0],nil]
    result
  end
.,.,

module_eval(<<'.,.,', 'parser.y', 81)
  def _reduce_29(val, _values, result)
     result = [val[0],val[2]]
    result
  end
.,.,

# reduce 30 omitted

module_eval(<<'.,.,', 'parser.y', 84)
  def _reduce_31(val, _values, result)
     result = val.join(' ')
    result
  end
.,.,

# reduce 32 omitted

# reduce 33 omitted

# reduce 34 omitted

module_eval(<<'.,.,', 'parser.y', 89)
  def _reduce_35(val, _values, result)
     result = val.join(' ')
    result
  end
.,.,

# reduce 36 omitted

# reduce 37 omitted

# reduce 38 omitted

# reduce 39 omitted

# reduce 40 omitted

# reduce 41 omitted

# reduce 42 omitted

# reduce 43 omitted

# reduce 44 omitted

# reduce 45 omitted

# reduce 46 omitted

module_eval(<<'.,.,', 'parser.y', 99)
  def _reduce_47(val, _values, result)
     result = val.join(' ')
    result
  end
.,.,

# reduce 48 omitted

# reduce 49 omitted

# reduce 50 omitted

module_eval(<<'.,.,', 'parser.y', 104)
  def _reduce_51(val, _values, result)
     result = val.join(' ')
    result
  end
.,.,

def _reduce_none(val, _values, result)
  val[0]
end

  end   # class Parser
end   # module Namae
